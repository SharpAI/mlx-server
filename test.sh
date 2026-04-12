OMNI_RES="🤖 Omni Output: <br/>The image is a close-up shot of a young person's face.<br/><br/>The audio is not provided."
cat <<HTML > test.html
<div class="response">$OMNI_RES</div>
HTML
cat test.html | grep "<div class=\"response\">"
