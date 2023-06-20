import os
import pandas as pd
from utils import get_dirname
from constants import CONCATENATOR, SEPARATOR


def construct_raw_data():
    data_dir = None  # Fill in path to raw data containing NOTEEVENTS, DIAGNOSES_ICD and PROCEDURES_ICD
    out_dir = get_dirname(__file__)
    load_args = {"header": 0, "compression": "gzip", "dtype": str, "on_bad_lines": "error"}

    # Load event data
    data = pd.read_csv(os.path.join(data_dir, "NOTEEVENTS.csv.gz"), **load_args)
    data.columns = [col.lower() for col in data.columns]
    # Filter out discharge summaries
    data = data[data["category"] == "Discharge summary"]
    # Load ICD9 diagnosis codes
    icd9_d_data = pd.read_csv(os.path.join(data_dir, "DIAGNOSES_ICD.csv.gz"), **load_args)
    icd9_d_data.columns = [col.lower() for col in icd9_d_data.columns]
    # Load ICD9 procedure codes
    icd9_p_data = pd.read_csv(os.path.join(data_dir, "PROCEDURES_ICD.csv.gz"), **load_args)
    icd9_p_data.columns = [col.lower() for col in icd9_p_data.columns]

    icd_code_dict = {}
    icd9_codes = pd.concat([icd9_d_data, icd9_p_data], ignore_index=True)
    for (subject_id, hadm_id), icd_df in icd9_codes.groupby(["subject_id", "hadm_id"]):
        codes = CONCATENATOR.join(sorted(icd_df["icd9_code"].astype(str).unique()))
        icd_code_dict[CONCATENATOR.join([subject_id, hadm_id])] = codes

    data["icd9_code"] = data.apply(lambda row: icd_code_dict.get(CONCATENATOR.join([row["subject_id"], row["hadm_id"]]), ""), axis=1)
    # Filter out empty ICD codes
    data = data[data["icd9_code"] != ""]
    data = data[data["icd9_code"] != "nan"]

    data["text"] = data["text"].apply(lambda text: text.replace("\n", "\t"))
    data["text"] = data["text"].apply(lambda text: text.replace(SEPARATOR, "<|>"))

    data.to_csv(os.path.join(out_dir, f"mimic.csv"), sep=SEPARATOR, index_label="idx", na_rep="nan")


if __name__ == "__main__":
    construct_raw_data()