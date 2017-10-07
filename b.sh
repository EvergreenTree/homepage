#!/bin/sh 
cd /Users/EvergreenFu/GitHub/homepage/
python2 jemdoc.py index.jemdoc;
python2 jemdoc.py gradthesis.jemdoc;
python2 jemdoc.py NDE.jemdoc;
python2 jemdoc.py stat.jemdoc;
python2 jemdoc.py pictures.jemdoc;

