#!/bin/sh
#
# file: nedc_rsynch.sh
#
# this script detects an error with rsync and automatically restarts
# rsync if an error occurs. It is used to keep rsync running over
# long periods of time when you might regularly lose your network conneciton.
#
# Usage: nedc_rsync.sh user@host:/path...
#
# Example:
#  nedc_rsync.sh nedc_tuh_eeg@www.isip.piconepress.com:~/data/tuh_eeg/ .
#

# set up an infinite loop
#
RC=1 
while [[ $RC -ne 0 ]]
do
    # display an informational message
    #
    echo "starting rsync..."

    # execute your rsync command
    #
    echo "starting rsync..."
    rsync -auxvL nedc@isip.piconepress.com:data/tuh_eeg_abnormal/v2.0.0/ . #rsync -auxv $1 .
    RC=$?

    # display an informational message and sleep for a bit
    #
    echo "done with rsync..."
    sleep 1

done

#
# exit gracefully
