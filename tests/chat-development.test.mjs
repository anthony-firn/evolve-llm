import test from 'node:test';
import assert from 'node:assert/strict';
import { wantsDevelopment, wantsSnakeStarter, developmentSummary } from '../web/chat-development.mjs';

test('Auto routes development requests and follow-ups while Chat and Build remain explicit',()=>{
 for(const text of ['Make me a Snake game','Could you build a calculator?','Fix the bug in app.js','Write a script to sort my file','pi Read README.txt','Read README.txt and save a summary','Save the output to result.json'])assert.equal(wantsDevelopment(text),true,text);
 for(const text of ['Hello','Write a haiku','Give me three tips for Rust code','Explain how to make a snake game','How do I write HTML?'])assert.equal(wantsDevelopment(text),false,text);
 assert.equal(wantsDevelopment('Make it blue','auto',true),true);
 assert.equal(wantsDevelopment('Try again','auto',true),true);
 assert.equal(wantsDevelopment('Thanks','auto',true),false);
 assert.equal(wantsDevelopment('Make me a Snake game','chat'),false);
 assert.equal(wantsDevelopment('Do this','build'),true);
});

test('Snake starters are only offered for creation requests',()=>{
 assert.equal(wantsSnakeStarter('Make me a snake game'),true);
 assert.equal(wantsSnakeStarter('Change the snake to blue'),false);
 assert.equal(wantsSnakeStarter('Make the snake blue'),false);
 assert.equal(wantsSnakeStarter('Make the snake game slower'),false);
 assert.equal(wantsSnakeStarter('Make another Snake game'),true);
 assert.equal(wantsSnakeStarter('Build a calculator'),false);
});

test('development summaries describe actual artifacts and preserve partial failures',()=>{
 const summary=developmentSummary({artifacts:[{path:'/workspace/snake.html'}],starterPath:'/workspace/snake.html',error:'Reply was truncated',errors:0},{ok:true});
 assert.match(summary,/included Snake starter/);assert.match(summary,/could not finish/);assert.match(summary,/truncated/);
 assert.match(developmentSummary({artifacts:[],errors:0},null),/did not produce an artifact/);
 assert.match(developmentSummary({artifacts:[],calls:1,errors:0,answer:'A summary of the read file.'},null),/A summary of the read file/);
 assert.match(developmentSummary({artifacts:[],errors:1},{ok:false}),/did not pass/);
 const deleted=developmentSummary({artifacts:[],removed:['/workspace/old.txt'],errors:0},null);
 assert.match(deleted,/Deleted old.txt/);assert.doesNotMatch(deleted,/No workspace files changed|did not produce/);
});

import { wantsBrowser } from '../web/chat-development.mjs';
test('Auto sends live-web requests to Pi with the browser and keeps file and chat requests as before',()=>{
 for(const text of ['Go to news.ycombinator.com and log in','Open https://example.com','What is on the front page of news.ycombinator.com?','Log in to my GitHub account in the browser','Click the login button on the website','Search wikipedia.org for Rust','Summarize https://example.com/post']){
  assert.equal(wantsBrowser(text),true,text);assert.equal(wantsDevelopment(text),true,text);
 }
 for(const text of ['Read README.txt and save a summary','Fix the bug in app.js','Open game.html','Make me a Snake game','How do I log in to github.com?','What is example.com?','Explain how websites work','Hello'])assert.equal(wantsBrowser(text),false,text);
 assert.equal(wantsDevelopment('Open https://example.com','chat'),false);
 for(const text of ['Yes, post it','I logged in','Go ahead','ok'])assert.equal(wantsDevelopment(text,'auto',true),true,text);
 assert.equal(wantsDevelopment('Yes','auto',false),false);
});
