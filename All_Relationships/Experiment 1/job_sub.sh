source set_env.sh
cd All_Relationships
bsub -o ~/sub-output.out -e ~/sub-error.out -q gpu -R "select[ngpus>0] rusage [ngpus_shared=1]" -n 1 python LSTM.py
