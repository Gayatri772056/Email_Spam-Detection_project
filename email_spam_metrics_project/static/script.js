
function predict(){
let text=document.getElementById("emailText").value;

fetch("/predict",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({text:text})
})
.then(res=>res.json())
.then(data=>{
document.getElementById("result").innerText=data.result;
document.getElementById("confidence").innerText=data.confidence+"%";
document.getElementById("bar").style.width=data.confidence+"%";
});
}
