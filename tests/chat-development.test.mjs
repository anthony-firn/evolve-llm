import test from 'node:test';
import assert from 'node:assert/strict';
import { wantsSnakeStarter, wantsBrowser, stepLabel } from '../web/chat-development.mjs';

test('Snake starters are only offered for creation requests',()=>{
 assert.equal(wantsSnakeStarter('Make me a snake game'),true);
 assert.equal(wantsSnakeStarter('Change the snake to blue'),false);
 assert.equal(wantsSnakeStarter('Make the snake blue'),false);
 assert.equal(wantsSnakeStarter('Make the snake game slower'),false);
 assert.equal(wantsSnakeStarter('Make another Snake game'),true);
 assert.equal(wantsSnakeStarter('Build a calculator'),false);
});

test('requests about a live website get the shared browser; files and questions do not',()=>{
 for(const text of ['Go to news.ycombinator.com and log in','Open https://example.com','What is on the front page of news.ycombinator.com?','Log in to my GitHub account in the browser','Click the login button on the website','Search wikipedia.org for Rust','Summarize https://example.com/post'])assert.equal(wantsBrowser(text),true,text);
 for(const text of ['Read README.txt and save a summary','Fix the bug in app.js','Open game.html','Make me a Snake game','How do I log in to github.com?','What is example.com?','Explain how websites work','Hello'])assert.equal(wantsBrowser(text),false,text);
});

test('steps read as plain actions',()=>{
 assert.equal(stepLabel('bash',{command:'cd app\ngit status'}),'Ran cd app; git status');
 assert.equal(stepLabel('edit',{path:'/home/user/app/README.md'}),'Edited README.md');
 assert.equal(stepLabel('write',{path:'snake.html'}),'Wrote snake.html');
 assert.equal(stepLabel('web_search',{query:'WebGPU'}),'Searched the web for “WebGPU”');
 assert.equal(stepLabel('open_url',{url:'https://example.com/a'}),'Read example.com/a');
 assert.equal(stepLabel('preview',{path:'/workspace/game.html'}),'Opened game.html');
 assert.equal(stepLabel('browser_click',{ref:3}),'Used the browser');
});
