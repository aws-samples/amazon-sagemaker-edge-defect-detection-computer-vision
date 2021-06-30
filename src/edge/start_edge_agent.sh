# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#!/bin/bash
if [ "$SM_EDGE_AGENT_HOME" == "" ]; then
    echo "You need to define the env var: SM_EDGE_AGENT_HOME"
    exit
fi

echo "SM_EDGE_AGENT_HOME: $SM_EDGE_AGENT_HOME"
AGENT_PID_FILE='/tmp/edge_agent.pid'
APP_PID_FILE='/tmp/edge_app.pid'
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if ! test -f "$AGENT_PID_FILE" || ! kill -0 $(cat $AGENT_PID_FILE) 2> /dev/null; then
    echo "Starting the agent"
    rm -f nohup.out /tmp/edge_agent
    nohup $SM_EDGE_AGENT_HOME/bin/sagemaker_edge_agent_binary -a /tmp/edge_agent -c $SM_EDGE_AGENT_HOME/conf/config_edge_device.json >> $SM_EDGE_AGENT_HOME/logs/agent.log 2>&1 &
    AGENT_PID=$!
    echo $AGENT_PID > $AGENT_PID_FILE
fi
echo "AGENT PID: $(cat $AGENT_PID_FILE)"

echo "Note: Please verify that the edge agent is running by using the command \"ps aux | grep [s]agemaker_edge_agent_binary\". In case you do not see any process running, please check the log file \"$SM_EDGE_AGENT_HOME/logs/agent.log\"".