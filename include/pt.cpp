#define CPU_ONLY
#include "caffe/caffe.hpp"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;
using namespace std;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

int main()
{
	NetParameter proto;
	ReadProtoFromBinaryFile("/home/pry/DeepCompression-caffe-master/examples/mnist/lenet_iter_5000.caffemodel", &proto);
	WriteProtoToTextFile(proto, "/home/pry/DeepCompression-caffe-master/netArgs.txt");
	return 0;
}


