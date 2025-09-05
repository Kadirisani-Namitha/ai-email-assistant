import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\kadir\Desktop\ai-email-assistant\data\Sample_Support_Emails_Dataset.csv")

# Keywords for urgent/important issues
keywords = ["urgent", "error", "support", "help", "critical", "downtime"]

# Filter emails based on subject OR body
filtered_df = df[
    df['subject'].str.contains('|'.join(keywords), case=False, na=False) |
    df['body'].str.contains('|'.join(keywords), case=False, na=False)
]

print("Filtered Important Emails:")
print(filtered_df[['sender', 'subject', 'sent_date']])

# Save results for later use
filtered_df.to_csv(r"C:\Users\kadir\Desktop\ai-email-assistant\data\urgent_emails.csv", index=False)
print("\nResults saved to urgent_emails.csv")
