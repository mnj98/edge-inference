sudo tc qdisc add dev eth0 root netem rate $1kbit
