#!/bin/bash

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python -m venv venv
else
  echo "Virtual environment already exists."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install fastapi CLI tools
echo "Installing FastAPI CLI tools..."
pip install fastapi-cli

echo "Setup complete! Virtual environment is now active."

# Đảm bảo source vào virtual environment trước khi hỏi người dùng
source venv/bin/activate

# Hỏi người dùng có muốn khởi động ứng dụng không
read -p "Bạn có muốn khởi động ứng dụng ngay bây giờ không? (y/n): " START_APP

if [ "$START_APP" = "y" ] || [ "$START_APP" = "Y" ]; then
  echo "Khởi động ứng dụng trên cổng 9000..."
  echo "Bạn có thể truy cập API docs tại: http://localhost:9000/docs"
  fastapi dev app/main.py --port 9000
else
  echo "Để khởi động ứng dụng sau, chạy lệnh: fastapi dev app/main.py --port 9000" sau khi source venv/bin/activate
fi