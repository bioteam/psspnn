"""
Ordered list of the 62 proteins used in Holley & Karplus (1989).

The first 48 are the training set; the last 14 are the test set.

Training set IDs come from table 1 of Kabsch & Sander (1983) FEBS Lett 155:179-182,
which is cited by Holley & Karplus as the source of their 62-protein benchmark.

Test set IDs are explicitly listed in Table 2 of Holley & Karplus (1989).

Note: Some PDB entries from the 1983 era have been superseded. The paper itself
flags 1HBL as obsolete (replaced by 1LH1 in Table 2). Download utilities handle
retrieval failures gracefully.
"""

# 48 training proteins (Kabsch & Sander 1983 table 1, first 48 entries)
# Order matters: this determines the train/test split.
TRAIN_IDS: list[str] = [
    "2RSA", "3RNS", "1RBX", "1PPT", "1SBT", "2PTN", "2SNS", "3FAB",
    "1LZM", "1SA0", "1HHO", "2HHB", "1MBD", "1MBA", "2MHB", "1CC5",
    "3CYT", "1PCY", "2B5C", "2C2C", "1C2C", "2FDN", "1FX1", "1FDX",
    "1HIP", "3ICB", "1BP2", "1ABE", "3APP", "2APP", "1APE", "1CPP",
    "2CGA", "1CHO", "3CPA", "2PKA", "1GCN", "2CRK", "1LH2", "4INS",
    "3INS", "1INS", "1PTI", "5PTI", "2OVO", "1ACX", "2ACT", "1LZ1",
]

# 14 test proteins (Holley & Karplus 1989, Table 2)
TEST_IDS: list[str] = [
    "1GPD", "4ADH", "2GRS", "2SOD", "1LH1", "1CRN",
    "1OVO", "2SSI", "1CTX", "1MLT", "1NXB", "2ADK", "1RHD", "2PAB",
]

ALL_IDS: list[str] = TRAIN_IDS + TEST_IDS


def get_split(split: str) -> list[str]:
    """Return PDB IDs for 'train', 'test', or 'all'."""
    if split == "train":
        return list(TRAIN_IDS)
    if split == "test":
        return list(TEST_IDS)
    if split == "all":
        return list(ALL_IDS)
    raise ValueError(f"Unknown split '{split}'; expected 'train', 'test', or 'all'.")
