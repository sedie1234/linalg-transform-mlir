/* fold-unit-extent-dims 심화 — unit 차원 collapse 애니메이션 + KaTeX init. 자체완결. */
(function(){
"use strict";
var C={bg:"#10131a",cell:"#2b3447",cellB:"#3a4456",unit:"#ffa657",real:"#3fb950",
       dim:"#9aa4b2",text:"#e6e9ef",accent:"#6f8cff",map:"#56d4dd"};
function txt(ctx,s,x,y,col,font,al){ctx.fillStyle=col;ctx.font=font||"12px ui-monospace,monospace";
  ctx.textAlign=al||"center";ctx.textBaseline="middle";ctx.fillText(s,x,y);}
function rr(ctx,x,y,w,h,r,fill,stroke,a){ctx.save();if(a!=null)ctx.globalAlpha=a;ctx.beginPath();
  r=Math.min(r,h/2,w/2);ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath();
  if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.stroke();}ctx.restore();}
function speedCtl(root){var s=root.querySelector(".spd"),v=root.querySelector(".spdv");
  function u(){if(v)v.textContent=(+s.value)+"×";}if(s){s.addEventListener("input",u);u();}
  return function(){return s?+s.value:1;};}

(function(){
  var root=document.getElementById("anim-fold"); if(!root)return;
  var cv=root.querySelector(".fd-canvas"),ctx=cv.getContext("2d");
  var sel=root.querySelector(".fd-preset"),fEl=root.querySelector(".fd-formula");
  var sp=speedCtl(root);
  var P=[
    {name:"1x5  — d0 unit (broadcast)", cells:5,
     dims:[{n:"d0",sz:"1",unit:true},{n:"d1",sz:"5",unit:false}],
     sb:"tensor<1x5xf32>", sa:"tensor<5xf32>", mb:"(d0,d1) -> (0, d1)", ma:"(d0,d1) -> (d1)"},
    {name:"5x1  — d1 unit", cells:5,
     dims:[{n:"d0",sz:"5",unit:false},{n:"d1",sz:"1",unit:true}],
     sb:"tensor<5x1xf32>", sa:"tensor<5xf32>", mb:"(d0,d1) -> (d0, 0)", ma:"(d0,d1) -> (d0)"},
    {name:"1x16x1  — d0,d2 unit (pad)", cells:16,
     dims:[{n:"d0",sz:"1",unit:true},{n:"d1",sz:"16",unit:false},{n:"d2",sz:"1",unit:true}],
     sb:"tensor<1x16x1xf32>", sa:"tensor<16xf32>", mb:"collapse_shape [[0,1,2]]", ma:"tensor<16xf32>"},
    {name:"1x?x1x1  — d0,d2,d3 unit (reduction)", cells:7,
     dims:[{n:"d0",sz:"1",unit:true},{n:"d1",sz:"?",unit:false},{n:"d2",sz:"1",unit:true},{n:"d3",sz:"1",unit:true}],
     sb:"tensor<1x?x1x1xf32>", sa:"tensor<?xf32>", mb:"4-loop (par,par,red,red)", ma:"1-loop (par)"}
  ];
  P.forEach(function(p,i){var o=document.createElement("option");o.value=i;o.textContent=p.name;sel.appendChild(o);});
  var pi=0, t=0, playing=false;

  function draw(){
    var p=P[pi];
    ctx.fillStyle=C.bg;ctx.fillRect(0,0,cv.width,cv.height);
    var W=cv.width, pad=60;
    var n=Math.min(p.cells,16), cw=Math.min(40,(W-2*pad)/n), cellW=cw*n;
    var ox=(W-cellW)/2;
    // 데이터 셀 (불변) — 하단
    var cy=cv.height-46;
    for(var i=0;i<n;i++){ rr(ctx,ox+i*cw+1,cy,cw-2,30,4,C.cell,C.cellB); }
    txt(ctx,"데이터 셀 — 불변 (collapse는 메타데이터)",ox,cy+44,C.dim,"11px system-ui","left");
    if(p.cells>16) txt(ctx,"… "+(p.cells)+"개",ox+cellW+6,cy+15,C.dim,"11px ui-monospace","left");
    // 차원 바 (위에서 아래로 스택)
    var by=26, bh=24, gap=8;
    for(var d=0;d<p.dims.length;d++){
      var dim=p.dims[d];
      var a=1, h=bh, yoff=0;
      if(dim.unit){ a=1-t; h=bh*(1-0.85*t); } // unit 바: fold 시 fade+shrink
      var col=dim.unit?C.unit:C.real, bg=dim.unit?"#2a2410":"#0f2417";
      if(a>0.02){
        rr(ctx,ox,by+yoff,cellW,h,6,bg,col,a);
        ctx.save();ctx.globalAlpha=a;
        txt(ctx,dim.n+": "+dim.sz+(dim.unit?"  (unit ✗)":""),ox+cellW/2,by+yoff+h/2,col,"600 12px ui-monospace");
        ctx.restore();
      }
      by += bh+gap;
    }
    // 라벨
    txt(ctx,"논리 차원 (shape)",ox,16,C.dim,"11px system-ui","left");
    // 화살표 (fold 진행)
    if(t>0.05){ txt(ctx,"collapse →",ox+cellW+8,30,C.accent,"600 12px ui-monospace","left"); }
    // formula
    var cur = t<0.5 ? p.sb : p.sa, curm = t<0.5 ? p.mb : p.ma;
    fEl.innerHTML = "shape: <b>"+p.sb+"</b> → <b style='color:var(--ok)'>"+p.sa+"</b>"
      + " &nbsp;|&nbsp; map: <b>"+p.mb+"</b> → <b style='color:var(--ok)'>"+p.ma+"</b>";
  }
  function play(){ t=0; draw(); playing=true;
    (function tick(){ if(!playing)return; t+=0.04; if(t>=1){t=1;draw();playing=false;return;} draw(); setTimeout(tick, 28/sp()); })(); }
  sel.addEventListener("change",function(){playing=false;pi=+sel.value;t=0;draw();});
  root.querySelector(".fd-play").addEventListener("click",play);
  root.querySelector(".fd-reset").addEventListener("click",function(){playing=false;t=0;draw();});
  draw();
})();
})();
