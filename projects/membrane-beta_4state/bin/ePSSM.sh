cd ../datasets/SingleFASTA
for entry in *.fasta
do
psiblast -query $entry -evalue 0.01 -db database/uniprot_sprot.fasta -num_iterations 3 -out_ascii_pssm ../PSSMasci/$entry.pssm -num_threads=8
echo "DONE!"
done
