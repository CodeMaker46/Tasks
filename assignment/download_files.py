import gdown

file_ids = {
    'Customers.csv': '1bu_--mo79VdUG9oin4ybfFGRUSXAe-WE',
    'Products.csv': '1IKuDizVapw-hyktwfpoAoaGtHtTNHfd0',
    'Transactions.csv': '1saEqdbBB-vuk2hxoAf4TzDEsykdKlzbF'
}

for filename, file_id in file_ids.items():
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, filename, quiet=False)
