import streamlit as st
import joblib

# Load the trained model
model = joblib.load("model.joblib")

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Define the categories
categories = ['IsAbusive', 'IsThreat', 'IsProvocative', 'IsObscene', 'IsHatespeech', 'IsRacist',
              'IsNationalist', 'IsSexist', 'IsHomophobic', 'IsReligious', 'IsHate', 'IsRadicalism']

# Define preprocess_text function
def preprocess_text(text):
    if text is not None:
        # Convert text to lowercase
        return text.lower()
    else:
        return ""  # Return an empty string if text is None

def predict_toxicity(comment):
    # Preprocess the comment
    preprocessed_comment = preprocess_text(comment)

    # Vectorize the comment using TF-IDF
    comment_tfidf = tfidf_vectorizer.transform([preprocessed_comment])

    # Predict toxicity
    prediction = model.predict(comment_tfidf)
    # Predict probabilities
    probabilities = model.predict_proba(comment_tfidf)

    # Return prediction and probabilities
    return prediction[0], probabilities[0]

# Main Streamlit function
def main():

    # Create radio buttons for navigation
    tab = st.sidebar.radio('Navigate to', ['HOME', 'THE IMPACT OF TOXIC COMMENTS', 'CYBERBULLYING', 'HARASSMENT', 'CHECK YOUR COMMENTS HERE!'])

    if tab == 'HOME':
        st.title('WHAT ARE TOXIC COMMENTS?')
        # Insert the image
        st.image("social-comments-removebg-preview.png", width=550)
        st.write("""            **Toxic comments** refer to any form of communication, typically found online, that is harmful, offensive, or abusive towards others. These comments can vary widely in content and intent but generally involve language or expressions that aim to demean, intimidate, or provoke emotional distress in the recipient. Toxic comments may include:

            1. **Abusive Language**: Comments containing insults, derogatory remarks, or offensive language targeted at an individual or group.
            
            2. **Threats**: Statements or expressions of intent to cause harm, fear, or violence towards others.
            
            3. **Provocative Remarks**: Comments designed to provoke anger, outrage, or conflict, often by intentionally stirring controversy or instigating arguments.
            
            4. **Obscenities**: Vulgar or sexually explicit language and content that may be offensive or inappropriate.
            
            5. **Hate Speech**: Comments that promote hatred, discrimination, or violence based on characteristics such as race, ethnicity, religion, gender, sexual orientation, or disability.
            
            6. **Racism**: Expressions of prejudice, discrimination, or bias based on race or ethnicity.
            
            7. **Nationalism**: Comments advocating for or glorifying one's own nationality or denigrating others based on their nationality or ethnic background.
            
            8. **Sexism**: Remarks that discriminate against or belittle individuals based on their gender or sex.
            
            9. **Homophobia**: Comments expressing hatred, prejudice, or discrimination towards individuals who identify as LGBTQ+.
            
            10. **Religious Intolerance**: Statements or attitudes that show disrespect or hostility towards individuals or groups based on their religious beliefs or practices.
            
            11. **Hate**: Comments driven by intense dislike or hostility towards individuals or groups, often based on personal prejudice or biases.
            
            12. **Radicalism**: Extreme or uncompromising views, ideologies, or beliefs that advocate for violence, extremism, or radical change.
            
            Toxic comments can have serious consequences, both for individuals and society as a whole. They can contribute to a hostile online environment, promote division and polarization, and cause psychological harm to those targeted. It's important to recognize and address toxic comments to foster a safer and more respectful online community.
            """)
        st.write('**Credits**: Inspired E- learning')

    elif tab == 'THE IMPACT OF TOXIC COMMENTS':
        st.title('THE IMPACT OF TOXIC COMMENTS ON A INDIVIDUAL AND SOCIETY')
        # Insert the image
        st.image("effects-removebg-preview.png", width=550)
        st.write("""
                      **The impact of toxic comments on individuals and society can be profound and far-reaching. Toxic comments can lead to psychological distress, emotional harm, and damage to self-esteem. They can also contribute to the spread of misinformation, polarization, and division within communities. Here's a brief overview of the impact:**

          **Impact on Individuals**:
          - **Psychological Distress**: Toxic comments can cause anxiety, depression, and stress in individuals who are targeted or exposed to them.
          - **Emotional Harm**: Individuals may experience feelings of anger, frustration, and helplessness as a result of toxic comments directed at them.
          - **Damage to Self-esteem**: Negative or derogatory comments can erode an individual's confidence and self-worth, leading to long-term self-esteem issues.

          **Impact on Society**:
          - **Polarization and Division**: Toxic comments contribute to polarization and division within society by fueling conflicts, fostering resentment, and promoting tribalism.
          - **Undermining Civil Discourse**: Toxic comments undermine civil discourse and constructive dialogue by discouraging open communication and respectful engagement.
          - **Spreading Misinformation**: Toxic comments may propagate false information, rumors, or conspiracy theories, leading to confusion and distrust in institutions and authorities.

          By addressing toxic comments and promoting respectful communication, we can mitigate their harmful effects and foster a more positive and inclusive online environment.

            """)
        st.write('**Credits**: Very Well Mind')

    elif tab == 'CYBERBULLYING':
        st.title('WHAT DOES CYBERBULLYING MEAN?')
        # Insert the image
        st.image("Cyberbullying-removebg-preview.png", width=550)
        st.write("""
              **Cyberbullying** is a form of harassment or bullying that occurs through digital devices such as smartphones, computers, or tablets. It involves the use of electronic communication to intimidate, threaten, or harm others. Here's a short overview:

            - **Definition**: Cyberbullying encompasses various behaviors, including sending abusive messages or emails, spreading rumors or false information online, creating fake profiles to harass someone, and sharing embarrassing or private photos or videos without consent.

            - **Impact**: Cyberbullying can have severe consequences for victims, including psychological distress, anxiety, depression, low self-esteem, and in extreme cases, suicidal thoughts or actions. It can also affect academic or professional performance and lead to social isolation.

            - **Prevention and Intervention**: To address cyberbullying, it's essential to promote digital citizenship and online safety education. Encouraging responsible online behavior, fostering empathy and kindness, and empowering bystanders to intervene can help prevent cyberbullying. Additionally, providing support services and resources for victims and implementing policies and procedures for reporting and addressing cyberbullying incidents are crucial steps in intervention.
   
            """)
        st.write('**Credits**: Inspired E- learning')
    elif tab == 'HARASSMENT':
        st.title('WHAT IS HARRASSMENT?')
        # Insert the image
        st.image("harras.png", width=550)
        st.write("""
                        **Harassment refers to any unwanted or unwelcome behavior that makes someone feel intimidated, offended, or distressed. It can occur in various forms, including verbal, physical, or digital harassment. Here's some information on harassment and how to address it:**

            **Types of Harassment**:
            - **Verbal Harassment**: Includes derogatory comments, insults, or threats made verbally or in writing.
            - **Physical Harassment**: Involves unwanted physical contact, gestures, or invasion of personal space.
            - **Digital Harassment (Cyberbullying)**: Occurs online through emails, social media, or messaging platforms, and may involve spreading rumors, threats, or sharing inappropriate content.

            **Effects of Harassment**:
            - Harassment can have serious psychological, emotional, and physical effects on victims, leading to anxiety, depression, and stress.
            - It can impact the victim's self-esteem, confidence, and overall well-being.
            - Harassment can also create a hostile or toxic environment, affecting productivity and morale in workplaces or communities.

            **Addressing Harassment**:
            - **Speak Up**: If you experience harassment, speak up and assertively communicate your boundaries to the perpetrator.
            - **Document Incidents**: Keep a record of the harassment incidents, including dates, times, and details of what happened.
            - **Seek Support**: Reach out to friends, family, or trusted individuals for support and guidance. Consider seeking help from counseling or support services.
            - **Report the Harassment**: Report the harassment to relevant authorities, such as human resources, school officials, or law enforcement, depending on the situation.
            - **Educate Others**: Raise awareness about harassment and its impact. Promote a culture of respect, tolerance, and inclusivity in your community or workplace.

            **Preventing Harassment**:
            - Promote education and awareness campaigns to prevent harassment and promote respectful behavior.
            - Implement policies and procedures for handling harassment complaints and providing support to victims.
            - Encourage bystander intervention and empower individuals to speak out against harassment when they witness it.

            By addressing harassment proactively and promoting a culture of respect and accountability, we can create safer and more inclusive environments for everyone.

            """)
        st.write('**Credits**: Inspired E- learning')
    elif tab == 'CHECK YOUR COMMENTS HERE!':
        st.title('TOXIC COMMENT CLASSIFIER')
        st.image("tcc-removebg-preview.png", width=550)
        # Input text area for user to enter comment
        comment = st.text_area('Enter your comment here:')
        submit_button = st.button('VALIDATE')
        if submit_button:
            if not comment.strip():
                st.warning('Please enter a comment.')
            else:
                # Make prediction when button is clicked
                result, probabilities = predict_toxicity(comment)
                if result == 1:
                    st.error('The content is inappropriate and hurtful.')
                    st.subheader('Percentage of toxicity according to categories:')
                    for i, category in enumerate(categories):
                        if i < len(probabilities):
                            st.write(f"{category}: {probabilities[i] * 100:.2f}%")
                else:
                    st.success('This feedback is constructive and respectful, contributing positively to the discussion.')

if __name__ == '__main__':
    main()
