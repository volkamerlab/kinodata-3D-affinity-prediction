condor_q
for FILE in $(find . -name "*$1*")
do
	echo $FILE
	cat $FILE
done
