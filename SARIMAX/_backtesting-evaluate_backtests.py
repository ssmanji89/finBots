import pandas as pd

def evaluate_trades(insights_csv_path):
    # Read the insights CSV
    insights_df = pd.read_csv(insights_csv_path)
    
    # Initialize empty columns for 'Outcome' and 'Signal_Strength'
    insights_df['Outcome'] = None
    insights_df['Signal_Strength'] = None
    
    # Loop through the DataFrame to evaluate each trade
    for index, row in insights_df.iterrows():
        # Initialize variables for easier access
        action = row['Action']
        entry_price = row['Entry_Price']
        exit_price = row['Exit_Price']
        
        # Evaluate Outcome based on action and price changes
        if action == 'Buy':
            if exit_price > entry_price:
                insights_df.loc[index, 'Outcome'] = 'Profit'
            elif exit_price < entry_price:
                insights_df.loc[index, 'Outcome'] = 'Loss'
            else:
                insights_df.loc[index, 'Outcome'] = 'Breakeven'
                
        elif action == 'Sell':
            if exit_price < entry_price:
                insights_df.loc[index, 'Outcome'] = 'Profit'
            elif exit_price > entry_price:
                insights_df.loc[index, 'Outcome'] = 'Loss'
            else:
                insights_df.loc[index, 'Outcome'] = 'Breakeven'
                
        # Evaluate Signal Strength based on technical indicators (simplified for this example)
        macd_condition = row['MACD_Condition']
        rsi_condition = row['RSI_Condition']
        
        if macd_condition and rsi_condition:
            insights_df.loc[index, 'Signal_Strength'] = 'Strong'
        elif macd_condition or rsi_condition:
            insights_df.loc[index, 'Signal_Strength'] = 'Moderate'
        else:
            insights_df.loc[index, 'Signal_Strength'] = 'Weak'
            
    # Save the evaluated DataFrame back to CSV
    insights_df.to_csv(insights_csv_path, index=False)
    
    return insights_df

# Assuming the insights CSV is stored at 'csv/insights.csv'
# Uncomment below to test the function
# evaluate_trades('csv/insights.csv')
