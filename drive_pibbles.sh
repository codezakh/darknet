#!/usr/bin/env expect
set images [glob -directory "predicted_images" -- "*.jpg"]
spawn ./darknet pibbles cfg/yolov3.cfg yolov3.weights
foreach image $images {
  expect "Enter Image Path:"
  send "$image\r"
}
send "data/dog.jpg\r"
