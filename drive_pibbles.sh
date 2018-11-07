#!/usr/bin/env expect
set images [glob -directory "predicted_images" -- "*.jpg"]
spawn ./darknet pibbles cfg/yolov3.cfg yolov3.weights
foreach image $images {
  puts "predicted_images/$image"
  expect "Enter Image Path:"
  send "predicted_images/$image"
  expect "Person"
}
send "data/dog.jpg\r"
