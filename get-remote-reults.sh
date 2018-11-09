rsync -av --exclude=*.npy --exclude=*.h5 aroth@40.68.7.162:~/grape-case/results ./remote-results
scp -r aroth@40.68.7.162:~/grape-case/results.log ./remote-results.log