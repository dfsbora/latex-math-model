#!/bin/bash
awk '
/^=== Temperature:/ { 
    if (NR > 1 && highest_bleu != "") { 
        print temp; 
        print highest_bleu_line; 
    }
    temp=$0; 
    highest_bleu=""; 
}
/BLEU Score:/ { 
    score=$3; 
    if (highest_bleu == "" || score+0 > highest_bleu+0) { 
        highest_bleu=score; 
        highest_bleu_line=$0; 
    } 
} 
END { 
    if (highest_bleu != "") { 
        print temp; 
        print highest_bleu_line; 
    }
}' eval_gpt2_prompt.txt
