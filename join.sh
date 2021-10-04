join -t$'\t' \
 <(zcat ukb39726.tab.gz | cut -f1,10120 | grep -v NA | sort -u) \
 <(cat \
  <(zcat gp_readv*toicd10.txt.gz | cut -d"|" -f2,9 | tr "|" "\t") \
  <(zcat hes_three_twodigit_withdup.txt.gz | cut -f1,5) \
  | sort -k1,1 | awk '{print $1"\t"substr($2,1,3)}') \
| sort -u
