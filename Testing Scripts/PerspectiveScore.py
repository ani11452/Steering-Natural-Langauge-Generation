from googleapiclient import discovery
from googleapiclient.errors import HttpError
import requests
import time

prev = time.time()

def get_perspective_score(text):
    API_KEY = "AIzaSyBhoNi1Om9gxeVIGcMdlLfTCeXYQgb8EU8"
    client = discovery.build(
      "commentanalyzer",
      "v1alpha1",
      developerKey=API_KEY,
      discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
      static_discovery=False,
    )

#    time.sleep(max(0, 1-(time.time()-prev)))
    time.sleep(1)
    analyze_request = {
      'comment': { 'text': f'{text}' },
      'requestedAttributes': {'TOXICITY': {}}
    }

    try:
        prev = time.time()
        response = client.comments().analyze(body=analyze_request).execute()
    except HttpError as error:
        print(f"An error occurred: {error}")
        return -1
    score = response['attributeScores']['TOXICITY']['summaryScore']['value']

    return score
