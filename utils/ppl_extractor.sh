#!/bin/bash
awk '
/^=== Temperature:/ { 
    if (NR > 1 && lowest_perplexity != "") { 
        print temp; 
        print lowest_perplexity_line; 
    }
    temp=$0; 
    lowest_perplexity=""; 
}
/Perplexity:/ { 
    score=$2; 
    if (lowest_perplexity == "" || score+0 < lowest_perplexity+0) { 
        lowest_perplexity=score; 
        lowest_perplexity_line=$0; 
    } 
} 
END { 
    if (lowest_perplexity != "") { 
        print temp; 
        print lowest_perplexity_line; 
    }
}' eval_gpt2_prompt.txt

