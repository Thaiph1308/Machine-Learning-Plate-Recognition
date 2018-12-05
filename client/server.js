//import the express module

var express = require('express');

//import body-parser
var bodyParser = require('body-parser');

//store the express in a variable 
var app = express();
var ejs = require('ejs')
app.use(express.static("public"));
app.set("view engine", "ejs");
app.set("views", "./views");
//configure body-parser for express
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());




app.use('/assets', express.static(__dirname + '/assets'))

//allow express to access our html (index.html) file
app.get('/index.html', function (req, res) {
  res.sendFile(__dirname + "/" + "index.html");
});

//route the GET request to the specified path, "/user". 
//This sends the user information to the path  

app.post('/process_post', function (req, res2) {
  response = {
    NoPlate: req.body.NoPlate,

  };
  //res.send("possst data")
  
  //this line is optional and will print the response on the command prompt
  //It's useful so that we know what infomration is being transferred 
  //using the server
  console.log(response);
  text = String(response.NoPlate);
  //convert the response in JSON format
  //res.end(JSON.stringify(response));
  var mongodb = require('mongodb');
  
  //We need to work with "MongoClient" interface in order to connect to a mongodb server.
  var MongoClient = mongodb.MongoClient;

  // Connection URL. This is where your mongodb server is running.
  var url = 'mongodb://giuxe:123456789bA@ds133113.mlab.com:33113/giuxe';

  // Use connect method to connect to the Server
  MongoClient.connect(url, { useNewUrlParser: true }, function (err, db) {
    if (err) {
      console.log('Unable to connect to the mongoDB server. Error:', err);
    }
    var dbo = db.db("giuxe");
    var myobj = { Plate: response };
    
    dbo.collection("xe").find({}).toArray(function (err, res) {
      if (err) throw err;
      check = false;

      for (i = 0; i < res.length; i++) {
        temp = JSON.stringify(String(res[i].ID));

        if (temp.indexOf(text) > 0) {
        
          var str1 = String(res[i].ID);
         
          var supHang1 = str1.split("_")[1];
          var supHang = str1.split("_")[2];
          var hang = supHang1.split("_")[0];
          var bss = supHang1.split("_")[1];
          
          var cot = supHang.split("-")[0];
          
          
          res2.render("output", { data:"Found motorbike at row "+hang +"; collum "+ cot});
          
          
         
         
          check = true;
        }
      }
      if (!check)
        
        res2.render("output", { data:"Not found"});
      db.close();
    })

  });
});

//This piece of code creates the server  
//and listens to the request at port 8888
//we are also generating a message once the 
//server is created
var server = app.listen(8888, function () {
  var host = server.address().address;
  var port = server.address().port;
  console.log("Example app listening at http://%s:%s", host, port);
});





