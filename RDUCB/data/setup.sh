#!/bin/bash
HOME_DIR = "~"
wget https://dl.dropboxusercontent.com/s/zsdapadpewlux37/fcnet.zip
unzip fcnet.zip
rm fcnet.zip
mv fcnet $HOME_DIR/fcnet
wget https://www.dropbox.com/s/kkxmrk747h9786o/mps.zip
unzip mps.zip
rm mps.zip
mv mps $HOME_DIR/mps
