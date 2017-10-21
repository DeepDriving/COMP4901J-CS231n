#!/bin/bash

apt-get update

apt-get install htop

apt-get install telnetd -y

apt-get install squid3 -y

apt-get istall squid -y

cd /etc/ssh

mv sshd_config sshd_config.old

touch sshd_config

echo "Port 22" >> sshd_config

echo "Port 443" >> sshd_config

echo "Protocol 2" >> sshd_config

echo "PermitRootLogin yes" >> sshd_config

echo "PermitEmptyPasswords yes" >> sshd_config

echo "PasswordAuthentication yes" >> sshd_config

echo "TCPKeepAlive yes"  >> sshd_config

echo "UseDNS yes" >> sshd_config

echo "Subsystem sftp /usr/lib/openssh/sftp-server" >> sshd_config

cd /etc/squid3

mv squid.conf squid.conf.old

touch squid.conf

echo "acl localhost src 127.0.0.1/255.255.255.255" >> squid.conf

echo "acl url1 url_regex -i 127.0.0.1" >> squid.conf

echo "acl url2 dstdomain .claro.com.br" >> squid.conf

echo "acl url3 dstdomain .claro.com.sv" >> squid.conf

echo "acl url4 dstdomain .speedtest.net" >> squid.conf

echo "acl url5 dstdomain .netclaro.com.br" >> squid.conf

echo "http_access allow localhost" >> squid.conf

echo "http_access allow url1" >> squid.conf

echo "http_access allow url2" >> squid.conf

echo "http_access allow url3" >> squid.conf

echo "http_access allow url4" >> squid.conf

echo "http_access allow url5" >> squid.conf

echo "http_access deny all" >> squid.conf

echo "cache deny all" >> squid.conf
 
echo "http_port 80" >> squid.conf

echo "http_port 8080" >> squid.conf

echo "visible_hostname  Phreaker Thyco" >> squid.conf

sleep 2s

echo "";
echo "Cadastro Efetuado com Sucesso !!!"

sleep 2s

echo "";

service ssh restart

service squid3 restart

echo Procedimentos Completos ! utilizar 127.0.0.1 como SSH IP
sleep 2s

cd /

echo > $HOME/.bash_history 
history -c
exit
