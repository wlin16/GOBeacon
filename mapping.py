import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import logging
from tqdm import tqdm
from pathlib import Path
import os
import argparse


header = {
    "version-11-5.string-db.org": "",
    "GET": "",
    "/network/7227.FBpp0082370": "",
    "https": "",
    "Accept": "",
    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7": "",
    "Accept-Encoding": "",
    "gzip,": "deflate, br",
    "Accept-Language": "",
    "en-GB,en;q=0.9,zh;q=0.8,en-US;q=0.7,zh-CN;q=0.6": "",
    "Cache-Control": "",
    "max-age=0": "",
    "Cookie": "",
    "string_login_credentials=_unassigned;": "_ga=GA1.1.778416608.1695314562; FPID=FPID2.2.ALSQXECYz5ixztL4PjvmbpqZSXlpeJrtE5Tzo%2BJp1Lc%3D.1695314562; string_db_ga_cid.11.5=bjLqKn5SG50g; string_db_userid.11.5=bAcQt0R9Pb5m; _pk_id.2.9c4d=b434be78a061df20.1700065539.; FPLC=aRqNIDF2ioZLZoRJbhORr1%2FsI05XPJnooTpkiH5VP1OqzJVz9MAXEduEaMLWSXRjNC%2B%2BPKfd8QlMqhLBiIhmbHkPou9PtcrTrSp6UvsHMeLret%2Bhf4ZpYrK%2F3r7%2FOw%3D%3D; _pk_ref.2.9c4d=%5B%22%22%2C%22%22%2C1701018770%2C%22https%3A%2F%2Fwww.uniprot.org%2F%22%5D; _pk_ses.2.9c4d=1; string_db_request_uri_hash.11.5=bD7qKyjYocah; string_db_taskid.11.5=bhorwP8paMmQ; _ga_P87S7MWQRL=GS1.1.1701018726.12.1.1701021044.0",
    "Sec-Ch-Ua": "",
    "\"Google": "Chrome\";v=\"119\", \"Chromium\";v=\"119\", \"Not?A_Brand\";v=\"24\"",
    "Sec-Ch-Ua-Mobile": "",
    "?0": "",
    "Sec-Ch-Ua-Platform": "",
    "\"macOS\"": "",
    "Sec-Fetch-Dest": "",
    "document": "",
    "Sec-Fetch-Mode": "",
    "navigate": "",
    "Sec-Fetch-Site": "",
    "none": "",
    "Sec-Fetch-User": "",
    "?1": "",
    "Upgrade-Insecure-Requests": "",
    "1": "",
    "User-Agent": "",
    "Mozilla/5.0": "(Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.ERROR)  # Set logger to capture ERROR level logs

# File handler - to capture error logs
file_handler = logging.FileHandler('./logging.log', mode='w')
file_handler.setLevel(logging.ERROR)  # Set level to ERROR for file handler
file_formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %A %H:%M:%S')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)


# Console handler - to display error logs
console_handler = logging.StreamHandler()
# Set level to ERROR for console handler
console_handler.setLevel(logging.ERROR)
console_formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


def log_error(uniprot_id, string_id, function_name, error_message):
    logger.error(
        f"UniProt ID: {uniprot_id}, STRING ID: {string_id}, Function: {function_name}, Error: {error_message}")


def get_target_links_for_STRING(base_url, url, uniprot_id, string_id, header):
    function_name = 'get_target_links_for_STRING'
    try:
        with requests.Session() as session:
            response = session.get(url, headers=header)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            target_links = [urljoin(base_url, link['href']) for link in soup.select("a.updateNonce")
                            if "string_interactions.tsv" in link['href'] or "protein_sequences.fa" in link['href']]
            return target_links
    except Exception as e:
        log_error(uniprot_id, string_id, function_name, str(e))
        return []


def parse_string_interaction(interaction_url, uniprot_id, string_id):
    function_name = 'parse_string_interaction'
    try:
        response = requests.get(interaction_url)
        response.raise_for_status()
        lines = response.content.decode().split("\n")
        protein1, protein2, score = [], [], []
        for line in lines[1:]:
            if line.strip():
                target = line.split("\t")
                protein1.append(target[2])
                protein2.append(target[3])
                score.append(target[-1])
        return pd.DataFrame({"protein1": protein1, "protein2": protein2, "score": score})
    except Exception as e:
        log_error(uniprot_id, string_id, function_name, str(e))
        return pd.DataFrame()


def parse_seq(seq_url, uniprot_id, string_id):

    function_name = 'parse_seq'
    try:
        response = requests.get(seq_url)
        response.raise_for_status()
        lines = response.content.decode()
        sequence_data = {}
        fasta_entries = lines.split('>')[1:]
        for entry in fasta_entries:
            lines = entry.strip().split("\n")
            identifier = lines[0].split()[0]
            sequence = "".join(lines[1:])
            sequence_data[identifier] = sequence
        return sequence_data
    except Exception as e:
        log_error(uniprot_id, string_id, function_name, str(e))
        return {}


def merge(interaction_df, seq_dict):
    try:
        interaction_df['seq1'] = interaction_df['protein1'].map(seq_dict)
        interaction_df['seq2'] = interaction_df['protein2'].map(seq_dict)
        return interaction_df
    except Exception as e:
        log_error("", "", "merge", str(e))
        return interaction_df


def read_file_list(dirctory):

    file_list = [file.stem for file in Path(f"{dirctory}").iterdir()]

    return set(file_list)

def read_error_file():
    error_df = pd.read_csv("error_ids.txt", sep="\t", names=["PID","STRING"])
    error_list = error_df["STRING"].tolist()
    return error_list

def check_error_files(df, output_dir):
    centre_node_list = df.index.tolist()
    string_centre_node = df['STRING'].tolist()
    error_list = []
    for centre_node_pid, string_id in tqdm(zip(centre_node_list, string_centre_node), total=len(centre_node_list)):
        dtype_spec = {
            'protein1': 'str',
            'protein2': 'str' 
        }
        ppi_csv = pd.read_csv(f"{output_dir}/{centre_node_pid}.csv", dtype=dtype_spec)
        flat_ppi_df = pd.concat([
                ppi_csv[['protein1', 'seq1']].rename(
                    columns={'protein1': 'protein', 'seq1': 'seq'}),
                ppi_csv[['protein2', 'seq2']].rename(
                    columns={'protein2': 'protein', 'seq2': 'seq'})
            ])
        flat_ppi_df = flat_ppi_df.drop_duplicates(
            subset=['protein'], keep="first")
        if string_id not in flat_ppi_df['protein'].tolist():
           error_list.append(string_centre_node)
    if error_list:
        rerun_df = df[df["STRING"].isin(error_list)]
        main(rerun_df, output_dir, base_url = "https://string-db.org")
    

def main(df, output_dir, base_url = "https://version-11-5.string-db.org"):
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        protein_id = index
        string_id = row['STRING']
        url = f"{base_url}/network/{string_id}"
        target_links = get_target_links_for_STRING(
            base_url,url, protein_id, string_id, header)

        for link in target_links:
            if "interactions" in link:
                inter_df = parse_string_interaction(
                    link, protein_id, string_id)
            elif "sequences" in link:
                seq_dict = parse_seq(link, protein_id, string_id)

        merged_df = merge(inter_df, seq_dict)
        merged_df.to_csv(f"{output_dir}/{protein_id}.csv", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process STRING database interactions for datasets.")
    parser.add_argument('-b', '--benchmark', required=True, help='Benchmark dataset name (CAFA3 or PDBch)')
    parser.add_argument('-d', '--filename', required=True, help='Dataset type (train_df.pkl or test_df.pkl)')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory for the crawled files')
    
    args = parser.parse_args()

    benchmark = args.benchmark
    filename = args.filename
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    data_file = f"data/{benchmark}/{filename}"

    df = pd.read_pickle(data_file)
    df.dropna(subset=['STRING'], inplace=True)
    main(df, output_dir)
    check_error_files(df, output_dir)

