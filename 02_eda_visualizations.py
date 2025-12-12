# 02_eda_visualizations.py
import pandas as pd
import matplotlib.pyplot as plt
import os

class RealEstateEDA:
    """
    Generate and save 20 EDA plots (files named eda_*.png)
    Reads processed_real_estate_data.csv
    """

    def __init__(self, processed_csv='processed_real_estate_data.csv', out_dir='eda_plots'):
        if not os.path.exists(processed_csv):
            raise FileNotFoundError(f"Processed CSV not found: {processed_csv}")
        self.df = pd.read_csv(processed_csv)
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir

    def save_fig(self, fig, name):
        path = os.path.join(self.out_dir, name)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")

    def run_all_eda(self):
        print("Running EDA and saving plots...")
        funcs = [
            self.plot_price_distribution,
            self.plot_size_distribution,
            self.plot_price_per_sqft_by_property_type,
            self.plot_price_vs_size_scatter,
            self.plot_price_per_sqft_outliers,
            self.plot_avg_price_per_sqft_by_state,
            self.plot_avg_price_by_city,
            self.plot_median_age_by_locality,
            self.plot_bhk_distribution_by_city,
            self.plot_top_localities_price_trend,
            self.plot_correlation_matrix,
            self.plot_schools_vs_price_per_sqft,
            self.plot_hospitals_vs_price_per_sqft,
            self.plot_price_by_furnished_status,
            self.plot_price_per_sqft_by_facing,
            self.plot_owner_type_counts,
            self.plot_availability_status_counts,
            self.plot_parking_vs_price,
            self.plot_amenities_vs_price_per_sqft,
            self.plot_transport_vs_price_or_investment
        ]

        for i, f in enumerate(funcs, 1):
            try:
                f()
                print(f"[{i}/20] Completed")
            except Exception as e:
                print(f"[{i}/20] Skipped due to error: {e}")

        print("EDA completed.")

    # Below: plotting implementations (each saves file)
    def plot_price_distribution(self):
        fig = plt.figure(figsize=(8,5))
        self.df['Price_in_Lakhs'].hist(bins=50)
        plt.title('Price Distribution (Lakhs)')
        plt.xlabel('Price (Lakhs)')
        plt.ylabel('Count')
        self.save_fig(fig, 'eda_1_price_distribution.png')

    def plot_size_distribution(self):
        fig = plt.figure(figsize=(8,5))
        self.df['Size_in_SqFt'].hist(bins=50)
        plt.title('Size Distribution (SqFt)')
        self.save_fig(fig, 'eda_2_size_distribution.png')

    def plot_price_per_sqft_by_property_type(self):
        if 'Property_Type' not in self.df.columns: return
        fig = plt.figure(figsize=(10,6))
        self.df.boxplot(column='Price_per_SqFt', by='Property_Type', rot=45)
        plt.title('Price per SqFt by Property Type')
        plt.suptitle('')
        self.save_fig(fig, 'eda_3_price_per_sqft_type.png')

    def plot_price_vs_size_scatter(self):
        fig = plt.figure(figsize=(8,6))
        sample = self.df.sample(n=min(2000, len(self.df)), random_state=42)
        plt.scatter(sample['Size_in_SqFt'], sample['Price_in_Lakhs'], alpha=0.4)
        plt.xlabel('Size_in_SqFt')
        plt.ylabel('Price_in_Lakhs')
        plt.title('Price vs Size (sample)')
        self.save_fig(fig, 'eda_4_price_vs_size.png')

    def plot_price_per_sqft_outliers(self):
        fig = plt.figure(figsize=(8,5))
        self.df['Price_per_SqFt'].plot.box()
        plt.title('Price per SqFt Boxplot')
        self.save_fig(fig, 'eda_5_price_per_sqft_outliers.png')

    def plot_avg_price_per_sqft_by_state(self):
        if 'State' not in self.df.columns: return
        series = self.df.groupby('State')['Price_per_SqFt'].mean().sort_values(ascending=False).head(20)
        fig = plt.figure(figsize=(10,6))
        series.plot.bar()
        plt.title('Avg Price per SqFt by State (Top 20)')
        self.save_fig(fig, 'eda_6_avg_price_per_sqft_by_state.png')

    def plot_avg_price_by_city(self):
        if 'City' not in self.df.columns: return
        series = self.df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(20)
        fig = plt.figure(figsize=(10,6))
        series.plot.bar()
        plt.title('Avg Price by City (Top 20)')
        self.save_fig(fig, 'eda_7_avg_price_by_city.png')

    def plot_median_age_by_locality(self):
        if 'Locality' not in self.df.columns: return
        series = self.df.groupby('Locality')['Age_of_Property'].median().sort_values().head(20)
        fig = plt.figure(figsize=(10,6))
        series.plot.bar()
        plt.title('Median Age by Locality (Top 20)')
        self.save_fig(fig, 'eda_8_median_age_by_locality.png')

    def plot_bhk_distribution_by_city(self):
        if 'City' not in self.df.columns: return
        fig = plt.figure(figsize=(10,6))
        self.df.groupby('City')['BHK'].value_counts().unstack(fill_value=0).sum(axis=1).head(20).plot.bar()
        plt.title('BHK Distribution (aggregated) by City (Top 20)')
        self.save_fig(fig, 'eda_9_bhk_distribution_by_city.png')

    def plot_top_localities_price_trend(self):
        if 'Locality' not in self.df.columns: return
        top = self.df.groupby('Locality')['Price_in_Lakhs'].median().sort_values(ascending=False).head(5).index
        fig = plt.figure(figsize=(10,6))
        for loc in top:
            series = self.df[self.df['Locality'] == loc].groupby('Year_Built')['Price_in_Lakhs'].median()
            series.plot(label=loc)
        plt.legend()
        plt.title('Price Trends for Top 5 Localities')
        self.save_fig(fig, 'eda_10_top_localities_price_trend.png')

    def plot_correlation_matrix(self):
        num = self.df.select_dtypes(include=['number'])
        fig = plt.figure(figsize=(10,8))
        corr = num.corr()
        plt.imshow(corr, cmap='RdYlBu', interpolation='nearest')
        plt.colorbar()
        plt.title('Correlation Matrix (numeric features)')
        self.save_fig(fig, 'eda_11_correlation_matrix.png')

    def plot_schools_vs_price_per_sqft(self):
        if 'Nearby_Schools' not in self.df.columns: return
        fig = plt.figure(figsize=(8,6))
        plt.scatter(self.df['Nearby_Schools'], self.df['Price_per_SqFt'], alpha=0.4)
        plt.title('Nearby Schools vs Price per SqFt')
        self.save_fig(fig, 'eda_12_schools_vs_price_per_sqft.png')

    def plot_hospitals_vs_price_per_sqft(self):
        if 'Nearby_Hospitals' not in self.df.columns: return
        fig = plt.figure(figsize=(8,6))
        plt.scatter(self.df['Nearby_Hospitals'], self.df['Price_per_SqFt'], alpha=0.4)
        plt.title('Nearby Hospitals vs Price per SqFt')
        self.save_fig(fig, 'eda_13_hospitals_vs_price_per_sqft.png')

    def plot_price_by_furnished_status(self):
        if 'Furnished_Status' not in self.df.columns: return
        fig = plt.figure(figsize=(8,6))
        self.df.boxplot(column='Price_in_Lakhs', by='Furnished_Status', rot=45)
        plt.title('Price by Furnished Status')
        plt.suptitle('')
        self.save_fig(fig, 'eda_14_price_by_furnished_status.png')

    def plot_price_per_sqft_by_facing(self):
        if 'Facing' not in self.df.columns: return
        fig = plt.figure(figsize=(10,6))
        self.df.boxplot(column='Price_per_SqFt', by='Facing', rot=45)
        plt.title('Price per SqFt by Facing')
        plt.suptitle('')
        self.save_fig(fig, 'eda_15_price_per_sqft_by_facing.png')

    def plot_owner_type_counts(self):
        if 'Owner_Type' not in self.df.columns: return
        fig = plt.figure(figsize=(6,4))
        self.df['Owner_Type'].value_counts().plot.bar()
        plt.title('Owner Type Counts')
        self.save_fig(fig, 'eda_16_owner_type_counts.png')

    def plot_availability_status_counts(self):
        if 'Availability_Status' not in self.df.columns: return
        fig = plt.figure(figsize=(6,4))
        self.df['Availability_Status'].value_counts().plot.bar()
        plt.title('Availability Status Counts')
        self.save_fig(fig, 'eda_17_availability_status_counts.png')

    def plot_parking_vs_price(self):
        if 'Parking_Space' not in self.df.columns: return
        fig = plt.figure(figsize=(8,6))
        plt.scatter(self.df['Parking_Space'], self.df['Price_in_Lakhs'], alpha=0.4)
        plt.title('Parking Space vs Price')
        self.save_fig(fig, 'eda_18_parking_vs_price.png')

    def plot_amenities_vs_price_per_sqft(self):
        if 'Has_Amenities' not in self.df.columns: return
        fig = plt.figure(figsize=(8,6))
        self.df.boxplot(column='Price_per_SqFt', by='Has_Amenities')
        plt.title('Amenities vs Price per SqFt')
        plt.suptitle('')
        self.save_fig(fig, 'eda_19_amenities_vs_price_per_sqft.png')

    def plot_transport_vs_price_or_investment(self):
        if 'Infrastructure_Score' not in self.df.columns: return
        fig = plt.figure(figsize=(8,6))
        plt.scatter(self.df['Infrastructure_Score'], self.df.get('Price_per_SqFt', self.df.get('Price_in_Lakhs')), alpha=0.4)
        plt.title('Infrastructure Score vs Price / Price_per_SqFt')
        self.save_fig(fig, 'eda_20_transport_investment.png')

if __name__ == "__main__":
    eda = RealEstateEDA()
    eda.run_all_eda()
