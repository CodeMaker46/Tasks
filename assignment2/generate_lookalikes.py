import pandas as pd
import json
from lookalike_model import LookalikeModel

def generate_lookalike_csv():
    print("Initializing Lookalike Model...")
    model = LookalikeModel()
    model.load_data()
    model.prepare_features()
    
    lookalike_map = {}
    for i in range(1, 21):
        customer_id = f'C{i:04d}'
        print(f"\nFinding similar customers for {customer_id}")
        
        try:
            _, recommendations = model.find_similar_customers(customer_id)
            similar_customers = [
                [rec['CustomerID'], float(f"{rec['Similarity']:.4f}")]
                for rec in recommendations
            ]
            lookalike_map[customer_id] = similar_customers
            
            print(f"Top 3 similar customers for {customer_id}:")
            for cust_id, score in similar_customers:
                print(f"  {cust_id}: {score}")
                
        except Exception as e:
            print(f"Error processing {customer_id}: {str(e)}")
            continue
    
    with open('Lookalike.csv', 'w') as f:
        f.write("CustomerID,SimilarCustomers\n")
        for cust_id, similar_list in lookalike_map.items():
            similar_str = json.dumps(similar_list)
            f.write(f"{cust_id},{similar_str}\n")
    
    print("\nLookalike recommendations saved to Lookalike.csv")
    print("\nSample of the generated map:")
    for cust_id, similar_list in list(lookalike_map.items())[:3]:
        print(f"\n{cust_id} -> {similar_list}")

if __name__ == "__main__":
    generate_lookalike_csv()
