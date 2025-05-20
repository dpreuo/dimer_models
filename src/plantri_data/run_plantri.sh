if [ ! -f /plantri_compiled ]; then
    gcc plantri55/plantri.c -o plantri_compiled
fi

for i in {30..40}
do
#    ./plantri_compiled -b $i -d some_graphs/graphs_out_$i

   ./plantri_compiled -b $i -d 1/1000000000 some_graphs/graphs_out_$i
done

# ./plantri_compiled -b 11 -d all_graphs/graphs_out_11