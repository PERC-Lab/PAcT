echo "Running Practice RQ1 Experiments"

conda activate exp

export EXPERIMENT=1
export EVAL=0

# Practice 
python3 final_exp_practice_collection.py
python3 final_exp_practice_other.py
python3 final_exp_practice_processing.py
python3 final_exp_practice_sharing.py

echo "Running Purpose RQ1 Experiments"
# Purpose
python3 final_exp_purpose_advertisement.py
python3 final_exp_purpose_analytics.py
python3 final_exp_purpose_functionality.py
python3 final_exp_purpose_other.py
