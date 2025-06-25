# Check coordinate ranges in your files
echo "Histone H3K4me1 coordinate range:"
awk '{print $2}' raw/E005-H3K4me1.narrowPeak | sort -n | head -5
awk '{print $2}' raw/E005-H3K4me1.narrowPeak | sort -n | tail -5

echo "DNase coordinate range:"  
awk '{print $2}' raw/E005-DNase.macs2.narrowPeak | sort -n | head -5
awk '{print $2}' raw/E005-DNase.macs2.narrowPeak | sort -n | tail -5

echo "DNase peaks per chromosome:"
cut -f1 raw/E003-DNase.macs2.narrowPeak | sort | uniq -c

echo "H3K4me1 peaks per chromosome:"  
cut -f1 raw/E003-H3K4me1.narrowPeak | sort | uniq -c

echo "Average H3K4me1 peak width:"
awk '{sum+=$3-$2; count++} END {print sum/count}' raw/E003-H3K4me1.narrowPeak

echo "Average DNase peak width:"
awk '{sum+=$3-$2; count++} END {print sum/count}' raw/E003-DNase.macs2.narrowPeak