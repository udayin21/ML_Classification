#!/bin/bash
# $1 is 1st argument
if [ $1 -eq 1 ] 
then
       if [ $2 -eq 1 ]
       then
       			python naivebayestest.py $3 $4
       elif [ $2 -eq 2 ]
       then 
       			python stemmingtest.py $3 $4
       elif [ $2 -eq 3 ]
       then 
       			python naivefeaturestest.py $3 $4			
       else
       			echo Unknown option
       fi										
elif [ $1 -eq 2 ]
then
		if [ $2 -eq 1 ]
       then
       			python digitclassifiertest.py $3 $4
       elif [ $2 -eq 2 ]
       then 
       			python digithelper.py $3
       			svm-scale -l 0 -u 1 libsvm_form.txt > libsvm_form_scaled.txt
       			svm-predict libsvm_form_scaled.txt q22model.model $4
       elif [ $2 -eq 3 ]
       then 
       			python digithelper.py $3
       			svm-scale -l 0 -u 1 libsvm_form.txt > libsvm_form_scaled.txt
       			svm-predict libsvm_form_scaled.txt q23model.model $4			
       else
       			echo Unknown option
       fi		
else 
     	echo Unknown option.
fi