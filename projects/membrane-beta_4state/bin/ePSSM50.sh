cd ../PSI-BLAST/SingleFASTA_50
for entry in *.fasta
do
psiblast -query $entry -evalue 0.01 -db /home/u2338/SU-project/projects/membrane-beta_4state/bin/uniprot_sprot.fasta -num_iterations 3 -out ../PSI_out_test/$entry -out_ascii_pssm ../PSSM_50test/$entry.pssm -num_threads=8
done


