"""
Author: Debora
Description: Performs the domain-specific evaluation over generated prompts saved in a csv file
Section 4.1, 4.2
"""

import pandas as pd
from utils.utils import EvaluatePrompt

if __name__ == "__main__":

    file_path = 'standard_prompts_output.csv'
    df = pd.read_csv(file_path)

    evaluator = EvaluatePrompt("Initialize")

    for index, row in df.iterrows():
        text = row['Generated text']
        evaluator.set_text(text)
        processed_output = evaluator.run_all()
        print(processed_output)

        for key, subdict in processed_output.items():
            df.at[index, f'{key}_total'] = subdict['total']
            df.at[index, f'{key}_score'] = subdict['score']

    print(df.head())

    df.to_csv('standard_prompts_output_evaluation.csv', index=False)
