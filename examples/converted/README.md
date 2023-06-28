
1. downloaded all the obj files
2. for i in in-*obj ; do o=$( echo ${i} | cut -f2- -d- ) ; ../../meshin-around.sh ${i} ${o} ; done 
3. for i in ../*png ; do o=$(basename ${i} | sed 's,-[^.]*\.,.,' ) ; cp -i ${i} ${o} ; done

