instructions: |
  ## Highlight references to fan's team or opponent in reddit NFL comments

prepend_instructions: true

edits:
  - name: in
    label: "In-group"
    enable_input: true
    color: orange      
    icon: fa-user 
    annotation:
      - name: conf
        label: "In-Confidence"
        question: "How confident are you with your annotation?"
        options: 
            - name: conf_1
              label: "No confidence"
            - name: conf_2
              label: "Little confidence"
            - name: conf_3
              label: "Somewhat confident"
            - name: conf_4
              label: "Very confident"
            - name: conf_5
              label: "Extremely confident"
      - name: comment_box
        label: "Comments box"
        question: "If you have any comments specific to this example, put them here:"
        options: textarea
        required: false

  - name: out
    label: "Out-group"
    enable_input: true
    color: "#964db0"
    icon: fa-person
    annotation:
      - name: conf
        label: "Out-Confidence"
        question: "How confident are you with your annotation?"
        options: 
            - name: conf_1
              label: "No confidence"
            - name: conf_2
              label: "Little confidence"
            - name: conf_3
              label: "Somewhat confident"
            - name: conf_4
              label: "Very confident"
            - name: conf_5
              label: "Extremely confident"     
      - name: comment_box
        label: "Comments box"
        question: "If you have any comments specific to this example, put them here:"
        options: textarea
        required: false

  - name: other
    label: "Other"
    enable_input: true
    color: teal     
    icon: fa-universal-access   
    annotation:
      - name: conf
        label: "Other-Confidence"
        question: "How confident are you with your annotation?"
        options: 
            - name: conf_1
              label: "No confidence"
            - name: conf_2
              label: "Little confidence"
            - name: conf_3
              label: "Somewhat confident"
            - name: conf_4
              label: "Very confident"
            - name: conf_5
              label: "Extremely confident"
      - name: comment_box
        label: "Comments box"
        question: "If you have any comments specific to this example, put them here:"
        options: textarea
        required: false

# For this tutorial, we disable the annotation component
disable:
 - upload
