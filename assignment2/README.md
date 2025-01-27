# Lookalike Model

This project implements a customer lookalike model that recommends similar customers based on their profiles and transaction history.

## Features

- Uses both customer profile and transaction data
- Calculates similarity scores using cosine similarity
- Recommends top 3 similar customers
- Takes into account:
  - Customer demographics (age, annual income, spending score)
  - Transaction history (number of transactions, total quantity, total spend)
  - Average transaction value

## Requirements

- Python 3.7+
- Required packages are listed in `requirements.txt`

## Installation

1. Clone the repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the model:
```bash
python lookalike_model.py
```

The script will:
1. Download the required data files
2. Initialize and train the lookalike model
3. Show an example recommendation for customer_id = 1

## Model Details

The model combines customer profile information with their transaction history to create a comprehensive feature set. Features are standardized using StandardScaler, and similarities are calculated using cosine similarity.

Key features used:
- Age
- Annual Income
- Spending Score
- Number of Transactions
- Total Quantity Purchased
- Total Amount Spent
- Average Transaction Value
