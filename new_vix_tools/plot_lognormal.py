import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load the CSV file
file_path = '/Users/preetam/Develop/ai-cookbook/testing_autogen/testing/dashboards/new_vix_tools/uploads/029ef953-f82e-40eb-af34-c8653d517ed3_external_loss_data.csv'
df = pd.read_csv(file_path)

# Convert the 'month__year_of_settlement' column to datetime.
if 'month__year_of_settlement' in df.columns:
    df['month__year_of_settlement'] = df['month__year_of_settlement'].astype(str).str.strip()
    try:
        df['settlement_date'] = pd.to_datetime(df['month__year_of_settlement'], format='%m/%Y', errors='coerce')
        if df['settlement_date'].isna().sum() > len(df)*0.5:
            df['settlement_date'] = pd.to_datetime(df['month__year_of_settlement'], infer_datetime_format=True, errors='coerce')
    except Exception as e:
        df['settlement_date'] = pd.to_datetime(df['month__year_of_settlement'], infer_datetime_format=True, errors='coerce')
else:
    raise ValueError('month__year_of_settlement column not found in CSV')

# Filter for last 6 months
today = pd.Timestamp.today()
six_months_ago = today - pd.DateOffset(months=6)
filtered_df = df[df['settlement_date'] >= six_months_ago]

# Get the loss data column; assuming column name is 'loss_amount___m_'
if 'loss_amount___m_' not in filtered_df.columns:
    raise ValueError('loss_amount___m_ column not found in the CSV file')

# Drop NaNs and ensure numeric type
loss_data = pd.to_numeric(filtered_df['loss_amount___m_'], errors='coerce').dropna()

# Remove non-positive values as lognormal is defined for >0
loss_data = loss_data[loss_data > 0]

if len(loss_data) == 0:
    raise ValueError('No valid loss data found for the last 6 months.')

# Fit lognormal distribution to the data by taking logarithm
log_data = np.log(loss_data)
mu, sigma = np.mean(log_data), np.std(log_data)

# Generate values for plotting the PDF
x = np.linspace(loss_data.min(), loss_data.max(), 1000)
# Manually compute the PDF for the lognormal distribution
pdf = (1/(x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu)**2)/(2 * sigma**2))

# Plot the histogram and the fitted lognormal distribution
plt.figure(figsize=(10, 6))
# Plot histogram with density normalization
plt.hist(loss_data, bins=30, density=True, color='skyblue', alpha=0.7, label='Loss Data')
# Plot the fitted lognormal PDF
plt.plot(x, pdf, color='red', lw=2, label='Lognormal Fit')

plt.title('Lognormal Distribution Fit on Loss Data for Last 6 Months')
plt.xlabel('Loss Amount (in millions)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()

# Save the chart with the provided filename
plt.savefig('029ef953-f82e-40eb-af34-c8653d517ed3_chart_1.png')
plt.close()

result = 'Chart created successfully and saved as 029ef953-f82e-40eb-af34-c8653d517ed3_chart_1.png'