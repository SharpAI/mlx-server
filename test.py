import sys, json

data = {"choices":[{"finish_reason":"stop","index":0,"message":{"content":"\nThe image is a black background with a white text overlay.","role":"assistant"}}]}

print('🤖 Omni Output:', data.get('choices',[{}])[0].get('message',{}).get('content', 'ERROR').replace('\n', '<br/>'))
