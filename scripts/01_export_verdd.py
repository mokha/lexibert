"""Export a MariaDB copy of the Ve'rdd database to parquet."""
import os
import pandas as pd, pymysql
O = os.environ.get("LEXIBERT_WORK", "work") + "/data/verdd"
con = pymysql.connect(host=os.environ.get("VERDD_HOST", "127.0.0.1"), port=int(os.environ.get("VERDD_PORT", 3306)),
                      user=os.environ["VERDD_USER"], password=os.environ["VERDD_PASSWORD"], database="verdd", charset="utf8mb4")
Q = {
    "lexemes": "select id, language_id lang, convert(lexeme using utf8mb4) lexeme, homoId, pos, contlex, type, checked, inflexType from manageXML_lexeme where deleted=0",
    "stems": "select lexeme_id, convert(text using utf8mb4) text, contlex, homoId, `order`, checked from manageXML_stem",
    "relations": "select id, type, lexeme_from_id f, lexeme_to_id t, checked, added_date from manageXML_relation where deleted=0 and lexeme_to_id is not null",
    "miniparadigm": "select lexeme_id, msd, wordform from manageXML_miniparadigm",
    "examples": "select lexeme_id, text, source from manageXML_example",
    "affiliations": "select lexeme_id, title, type from manageXML_affiliation",
    "lexeme_metadata": "select lexeme_id, type, text from manageXML_lexememetadata",
}
for name, sql in Q.items():
    df = pd.read_sql(sql, con)
    df.to_parquet(f"{O}/{name}.parquet", index=False)
    print(name, len(df), flush=True)
