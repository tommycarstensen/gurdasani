cut -f2 multimorbidity_icd10.raw.txt | grep -v ^Exclude | tr "," "\n" | grep [0-9] | awk '{print $1}' | sort -u > multimorbidity_icd10.txt
