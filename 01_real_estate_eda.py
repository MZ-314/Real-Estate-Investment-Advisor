# 01_preprocessing.py
import pandas as pd
import numpy as np
import os

class RealEstatePreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        self.df = pd.read_csv(file_path)
        print(f"Loaded {file_path} -> shape: {self.df.shape}")

    def explore_data(self, n=5):
        print("First rows:")
        print(self.df.head(n))
        print("\nMissing values:")
        print(self.df.isnull().sum())
        print("\nDtypes:")
        print(self.df.dtypes)

    def handle_missing_values(self):
        # numeric -> median; categorical -> mode
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col].fillna(self.df[col].median(), inplace=True)
            else:
                # if column totally empty, fill with placeholder
                if self.df[col].dropna().empty:
                    self.df[col].fillna("Unknown", inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        print("Missing values handled.")

    def feature_engineering(self, current_year=2024):
        df = self.df

        # 1. Price_per_SqFt
        if {'Price_in_Lakhs', 'Size_in_SqFt'}.issubset(df.columns) and 'Price_per_SqFt' not in df.columns:
            # avoid division by zero
            df['Size_in_SqFt'] = df['Size_in_SqFt'].replace(0, np.nan).fillna(df['Size_in_SqFt'].median())
            df['Price_per_SqFt'] = (df['Price_in_Lakhs'] * 100000) / df['Size_in_SqFt']
            print("Created: Price_per_SqFt")

        # 2. Age_of_Property
        if 'Year_Built' in df.columns and 'Age_of_Property' not in df.columns:
            df['Age_of_Property'] = current_year - df['Year_Built']
            df['Age_of_Property'] = df['Age_of_Property'].clip(lower=0)  # guard
            print("Created: Age_of_Property")

        # 3. Total_Nearby_Facilities
        nearby_cols = [c for c in df.columns if c.lower().startswith('nearby_') or 'Nearby' in c]
        if nearby_cols:
            df[nearby_cols] = df[nearby_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            df['Total_Nearby_Facilities'] = df[nearby_cols].sum(axis=1)
            print(f"Created: Total_Nearby_Facilities from {len(nearby_cols)} columns")

        # 4. Property_Category
        if 'BHK' in df.columns:
            df['Property_Category'] = pd.cut(
                df['BHK'], bins=[0,1,2,3,100], labels=['Studio','Small','Medium','Large']
            )
            print("Created: Property_Category")

        # 5. Is_New_Property
        if 'Age_of_Property' in df.columns:
            df['Is_New_Property'] = (df['Age_of_Property'] <= 5).astype(int)
            print("Created: Is_New_Property")

        # 6. Has_Amenities
        if 'Amenities' in df.columns:
            df['Has_Amenities'] = df['Amenities'].notna().astype(int)
            print("Created: Has_Amenities")

        # 7. Infrastructure_Score (ordinal mapping + numeric fallback)
        infra_cols = ['Public_Transport_Accessibility', 'Nearby_Schools', 'Nearby_Hospitals']
        valid = []
        ordinal_map = {
            "poor": 1, "below average": 2, "average": 3,
            "good": 4, "very good": 5, "excellent": 6,
            "low": 1, "medium": 3, "high": 5
        }

        for col in infra_cols:
            if col in df.columns:
                # normalize text
                if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                    # map common textual values (case-insensitive)
                    df[col] = df[col].astype(str).str.strip().str.lower()
                    df[col] = df[col].map(ordinal_map).fillna(pd.to_numeric(df[col], errors='coerce'))
                # numeric fallback
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(df[col].median(), inplace=True)
                valid.append(col)

        if valid:
            df['Infrastructure_Score'] = df[valid].mean(axis=1)
            print(f"Created: Infrastructure_Score using {len(valid)} fields")

        self.df = df

    def create_targets(self, growth_rate=0.08, years=5):
        df = self.df
        # Regression target
        if 'Price_in_Lakhs' in df.columns:
            df['Future_Price_5Y'] = df['Price_in_Lakhs'] * ((1 + growth_rate) ** years)
            print("Created: Future_Price_5Y (projection)")

        # Classification target (multi-factor)
        conditions = []
        if 'Price_per_SqFt' in df.columns:
            conditions.append(df['Price_per_SqFt'] <= df['Price_per_SqFt'].median())
        if 'BHK' in df.columns:
            conditions.append(df['BHK'] >= 2)
        if 'Age_of_Property' in df.columns:
            conditions.append(df['Age_of_Property'] < 20)
        if 'Infrastructure_Score' in df.columns:
            conditions.append(df['Infrastructure_Score'] >= df['Infrastructure_Score'].median())

        if conditions:
            # sum of boolean Series
            score = sum(conditions)
            df['Good_Investment'] = (score >= 2).astype(int)
            print(f"Created: Good_Investment (count = {df['Good_Investment'].sum()})")

        self.df = df

    def detect_outliers(self, numeric_subset=None):
        df = self.df
        if numeric_subset is None:
            numeric_subset = df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_indices = set()
        for col in numeric_subset:
            # skip trivial columns
            if df[col].nunique() <= 1:
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            idx = df[(df[col] < lower) | (df[col] > upper)].index
            outlier_indices.update(idx)

        print(f"Outliers identified: {len(outlier_indices)} rows")
        if outlier_indices:
            self.df = df.drop(list(outlier_indices)).reset_index(drop=True)
            print(f"Dropped {len(outlier_indices)} rows due to outliers. New shape: {self.df.shape}")
        else:
            print("No outliers dropped.")

    def encode_categoricals(self):
        # Keep encoders minimal — label encode low-cardinality categorical columns
        df = self.df.copy()
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if df[col].nunique() <= 100:  # safe threshold
                df[col] = df[col].astype(str)
                df[col] = df[col].fillna('Unknown')
                df[col] = pd.Categorical(df[col]).codes
        self.df = df
        print("Categorical encoding (ordinal) applied where safe.")

    def save(self, out_path='processed_real_estate_data.csv'):
        self.df.to_csv(out_path, index=False)
        print(f"Saved processed data to {out_path}")

if __name__ == "__main__":
    pre = RealEstatePreprocessor('india_housing_prices.csv')
    pre.explore_data()
    pre.handle_missing_values()
    pre.feature_engineering()
    pre.create_targets()
    pre.detect_outliers()  # call after creating targets to avoid dropping important rows prematurely
    pre.encode_categoricals()
    pre.save()
