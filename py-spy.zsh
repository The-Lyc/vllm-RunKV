for i in 1 2 3; do
    py-spy dump --pid 2282568 > /tmp/spy_$i.txt
    sleep 3
done
diff /tmp/spy_1.txt /tmp/spy_2.txt
diff /tmp/spy_2.txt /tmp/spy_3.txt
