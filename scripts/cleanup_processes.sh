pkill -f "python -m dynamo.frontend" || true
pkill -f "python -m dynamo.vllm" || true
pkill -f "VLLM::EngineCore" || true
pkill -f "aiperf profile" || true
sleep 3
