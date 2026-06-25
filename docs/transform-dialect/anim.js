/* transform dialect intro — "schedule 이 payload 에 적용" 애니메이션.
   같은 payload(matmul) 가 schedule(A: tile_using_for / C: tile_using_forall)에
   따라 다른 loop 구조로 타일링되는 과정을 보인다. 자체완결. */
(function(){
"use strict";
var root=document.getElementById("anim-transform"); if(!root) return;
var cv=root.querySelector(".tf-canvas"), ctx=cv.getContext("2d");
var sel=root.querySelector(".tf-sched"), fEl=root.querySelector(".tf-formula");
var sld=root.querySelector(".tf-spd"), spv=root.querySelector(".tf-spdval");
function sp(){ return sld?+sld.value:1; }
if(sld){ sld.addEventListener("input",function(){ if(spv)spv.textContent=(+sld.value)+"×"; }); if(spv)spv.textContent=sp()+"×"; }

var C={bg:"#10131a",text:"#e6e9ef",dim:"#9aa4b2",box:"#1b2330",boxB:"#3a4456",
       match:"#6f8cff",forc:"#6f8cff",forall:"#3fb950",axis:"#56d4dd",grid:"#2b3650"};
var SCHED={
  A:{name:"transform.structured.tile_using_for  tile_sizes [32,32,64]",rows:4,cols:2,kind:"for",
     done:"= scf.for ×3 순차 타일 nest (inner 32×32 matmul)",col:C.forc},
  C:{name:"transform.structured.tile_using_forall  num_threads [2,2]",rows:2,cols:2,kind:"forall",
     done:"= scf.forall 2×2 병렬 (parallel_insert_slice)",col:C.forall},
};
var CAP=[
 "payload: linalg.matmul (128×64 iteration space). 아직 schedule 적용 전.",
 "transform.structured.match → handle %mm 이 matmul 을 가리킴. (변환이 아니라 '지목')",
 "tile 적용 — iteration space 를 블록으로 쪼갬. (interpreter 가 payload 를 실제로 재작성)",
 null, // = SCHED.done
];

function rr(x,y,w,h,r,fill,stroke,lw){ ctx.beginPath(); r=Math.min(r,h/2,w/2);
  ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath();
  if(fill){ctx.fillStyle=fill;ctx.fill();} if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=lw||1.5;ctx.stroke();} }
function txt(s,x,y,col,font,al){ ctx.fillStyle=col;ctx.font=font||"13px ui-monospace,monospace";
  ctx.textAlign=al||"center";ctx.textBaseline="middle";ctx.fillText(s,x,y); }

var BX=300, BY=70, BW=300, BH=210; // iteration space rect

function draw(phase,prog){
  var S=SCHED[sel.value];
  ctx.fillStyle=C.bg; ctx.fillRect(0,0,cv.width,cv.height);
  // 단계 점
  for(var p=0;p<4;p++){ ctx.fillStyle=p<=phase?S.col:C.boxB; ctx.beginPath(); ctx.arc(24+p*16,22,5,0,7); ctx.fill(); }
  txt("schedule 적용 단계 "+(phase+1)+"/4", 24+4*16+12, 22, C.dim, "12px ui-monospace,monospace","left");
  // 축 라벨
  txt("M=128", BX-30, BY+BH/2, C.axis, "12px ui-monospace,monospace","right");
  txt("N=64", BX+BW/2, BY-16, C.axis, "12px ui-monospace,monospace","center");
  // 박스
  var matched = phase>=1;
  var bd = matched ? C.match : C.boxB;
  rr(BX,BY,BW,BH,8,C.box,bd, matched?2.4:1.5);
  // tile 그리드 (phase>=2)
  if(phase>=2){
    var pr = phase===2 ? prog : 1;
    var rows=S.rows, cols=S.cols;
    ctx.strokeStyle=S.col; ctx.lineWidth=1.4; ctx.globalAlpha=0.85*pr;
    for(var i=1;i<rows;i++){ var y=BY+BH*i/rows; ctx.beginPath(); ctx.moveTo(BX,y); ctx.lineTo(BX+BW,y); ctx.stroke(); }
    for(var j=1;j<cols;j++){ var x=BX+BW*j/cols; ctx.beginPath(); ctx.moveTo(x,BY); ctx.lineTo(x,BY+BH); ctx.stroke(); }
    // 타일 채움(살짝)
    ctx.globalAlpha=0.10*pr; ctx.fillStyle=S.col;
    for(var r2=0;r2<rows;r2++)for(var c2=0;c2<cols;c2++){
      ctx.fillRect(BX+BW*c2/cols+2,BY+BH*r2/rows+2,BW/cols-4,BH/rows-4);
    }
    ctx.globalAlpha=1;
    if(S.kind==="forall") txt("∥ 병렬", BX+BW/2, BY+BH/2, S.col, "700 14px ui-monospace,monospace");
  }
  // payload 라벨
  txt("linalg.matmul", BX+BW/2, phase>=2? BY+BH+22 : BY+BH/2-8, C.text, "600 14px ui-monospace,monospace");
  if(phase<2) txt("ins(A,B) outs(C)", BX+BW/2, BY+BH/2+14, C.dim, "12px ui-monospace,monospace");
  // match 핸들
  if(phase>=1){
    ctx.strokeStyle=C.match; ctx.lineWidth=1.6; ctx.setLineDash([5,4]);
    ctx.beginPath(); ctx.moveTo(BX+BW+10, BY+10); ctx.lineTo(BX+BW+90, BY-10); ctx.stroke(); ctx.setLineDash([]);
    txt("handle %mm", BX+BW+95, BY-14, C.match, "12px ui-monospace,monospace","left");
  }
  // schedule 텍스트
  txt(S.name, cv.width/2, cv.height-40, S.col, "12px ui-monospace,monospace","center");
  fEl.textContent = (phase===3 ? S.done : CAP[phase]);
}

var phase=0, playing=false, last=0, el=0; var DW=1300;
function loop(ts){
  if(!playing) return;
  if(!last) last=ts; var dt=(ts-last)*sp(); last=ts; el+=dt;
  draw(phase, Math.min(1, el/650));
  if(el>=DW){ if(phase>=3){ playing=false; draw(3,1); return; } phase++; el=0; }
  requestAnimationFrame(loop);
}
function play(){ phase=0; playing=true; last=0; el=0; requestAnimationFrame(loop); }
root.querySelector(".tf-play").addEventListener("click",play);
root.querySelector(".tf-reset").addEventListener("click",function(){ playing=false; phase=0; draw(0,1); });
sel.addEventListener("change",function(){ playing=false; phase=0; draw(0,1); });
draw(0,1);
})();
