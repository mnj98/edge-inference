sudo tc qdisc del dev eth0 root netem
sudo tc qdisc add dev eth0 root netem rate $1kbit loss $2 delay $3ms $4ms distribution normal