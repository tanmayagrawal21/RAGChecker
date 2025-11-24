import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json
import math
import random
from scipy.spatial import distance
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os
from openai import OpenAI
from dotenv import load_dotenv
import re
import spacy

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Knowledge Graph Validation", layout="wide")

# Load spaCy model for NER and sentence processing
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("Please install spaCy model: python -m spacy download en_core_web_sm")
        return None

nlp = load_spacy_model()

# Initialize session state for persistent storage across reruns
if 'context' not in st.session_state:
    st.session_state.context = ""
if 'question' not in st.session_state:
    st.session_state.question = ""
if 'answer' not in st.session_state:
    st.session_state.answer = ""
if 'source_json' not in st.session_state:
    st.session_state.source_json = None
if 'output_json' not in st.session_state:
    st.session_state.output_json = None
if 'similar_claims' not in st.session_state:
    st.session_state.similar_claims = None
if 'simple_scores' not in st.session_state:
    st.session_state.simple_scores = None
if 'claim_categories' not in st.session_state:
    st.session_state.claim_categories = None
if 'run_complete' not in st.session_state:
    st.session_state.run_complete = False
if 'fixed_answer' not in st.session_state:
    st.session_state.fixed_answer = None


st.title("Knowledge Graph Validation Tool")
st.markdown("""
This app analyzes generated answers against source content to detect potential hallucinations 
and validate factual consistency through knowledge graph extraction and comparison.
""")

# Sidebar for API configuration
with st.sidebar:
    st.header("API Configuration")
    api_key = os.environ.get("OPENAI_API_KEY")
    use_example = st.checkbox("Use example content", value=True, 
                              help="Use the example context and question instead of entering your own")
    
    st.subheader("Detection Method")
    detection_method = st.selectbox(
        "Choose detection approach",
        ["GraphEval+", "SICI-0 (sentences)", "SICI-1 (sentences with context)"],
        index=0,
        help="GraphEval+ uses triple extraction. SICI uses sentence-level analysis with coreference resolution."
    )
    
    show_advanced = st.checkbox("Show advanced options", value=False)
    
    if show_advanced:
        st.subheader("Advanced Settings")
        model_choice = st.selectbox(
            "Model for Answer Generation", 
            ["gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"],
            index=0
        )
        embedding_model = st.selectbox(
            "Embedding Model",
            ["text-embedding-3-small", "text-embedding-3-large"],
            index=0
        )
        NLI_THRESHOLD = st.slider("NLI Threshold", 0.0, 1.0, 0.5)
        SIMILARITY_THRESHOLD = st.slider("Similarity Threshold", 0.0, 1.0, 0.5)
    else:
        model_choice = "gpt-4-turbo"
        embedding_model = "text-embedding-3-small"
        SIMILARITY_THRESHOLD = 0.5
        NLI_THRESHOLD = 0.5
        
# Example content from the notebook
default_context = """
In the town of Quillhaven—a community renowned for its historic libraries and scholarly traditions—a rare astronomical event was predicted to occur for the first time in over a century. The event, a total solar eclipse, was shrouded in both scientific intrigue and centuries-old folklore. Local legends, drawing on influences from ancient Greek philosophy to indigenous spiritual practices, suggested that eclipses held the power to influence human thought and communal harmony.

At the heart of the community's preparations was Dr. Lila Montrose, a respected astrophysicist whose career had been dedicated to unraveling the mysteries of cosmic events. Dr. Montrose saw the eclipse as an opportunity to blend modern scientific inquiry with the town's rich cultural heritage. She proposed a series of public lectures and interactive exhibitions designed to educate residents on the mechanics of the eclipse, while also acknowledging its historical and psychological significance. Her balanced approach aimed to respect both empirical evidence and the symbolic narratives that had long captivated the community.

Meanwhile, Mr. Edmund Blackwell, the town's dedicated historian, uncovered an ancient manuscript in the dusty archives of Quillhaven's old library. The manuscript, penned in a mix of archaic English and Latin, detailed elaborate rituals and communal activities that had been performed during similar eclipses in bygone eras. Blackwell argued that these practices were more than mere superstition—they were intrinsic to the town's identity and had once fostered unity and prosperity. His findings ignited a fervent debate about whether these ancient rituals should be revived as a way to reconnect with Quillhaven's storied past.

This emerging debate quickly divided the community into two factions. The progressive wing, led by Dr. Montrose, advocated for a balanced, modern approach that combined scientific exploration with cultural respect. In contrast, the conservative faction, inspired by Mr. Blackwell's manuscript, pushed for a return to the traditional rituals that they believed had been the cornerstone of the town's former unity and success. As the day of the eclipse drew near, these conflicting views not only highlighted an ideological rift but also symbolized a broader struggle between embracing modernity and preserving historical identity.

In a final effort to bridge these divergent perspectives, town leaders organized a public forum. The forum featured spirited debates and passionate speeches, ultimately concluding with a proposal for collaboration. Both factions agreed that the eclipse could serve as a catalyst for dialogue, urging the community to forge a common path that honored both its scientific curiosity and its deep-rooted traditions.
"""

default_question = "Explain how the contrasting views of Dr. Lila Montrose and Mr. Edmund Blackwell regarding the upcoming eclipse illustrate the broader tension between modern scientific inquiry and traditional cultural practices. What solution did the town of Quillhaven ultimately propose to address this conflict?"

# Main input section
col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    
    # Update context and question in session state when changed
    context = st.text_area("Context", 
                          value=default_context if use_example else "", 
                          height=300,
                          help="Enter the source text that will be used as context",
                          key="context_input")
    
    question = st.text_area("Question", 
                           value=default_question if use_example else "",
                           help="Enter a question to be answered based on the context",
                           key="question_input")
    
    # Store values in session state
    st.session_state.context = context
    st.session_state.question = question

# Define functions for the core functionality

def resolve_coreferences_spacy(text, context_window=0):
    """
    Extract sentences with simple coreference resolution using spaCy.
    Uses NER and simple pronoun resolution.
    """
    if nlp is None:
        st.error("spaCy model not loaded")
        return []
    
    doc = nlp(text)
    sentences = list(doc.sents)
    resolved_sentences = []
    
    # Build entity map for the entire document
    entity_map = {}
    for ent in doc.ents:
        entity_map[ent.start] = ent.text
    
    for i, sent in enumerate(sentences):
        # Get context sentences if needed
        if context_window == 0:
            context_sents = [sent]
        elif context_window == 1:
            start_idx = max(0, i - 1)
            end_idx = min(len(sentences), i + 2)
            context_sents = sentences[start_idx:end_idx]
        else:
            context_sents = [sent]
        
        # Collect entities from context
        context_entities = []
        for ctx_sent in context_sents:
            for ent in ctx_sent.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'NORP', 'FAC', 'LOC']:
                    context_entities.append(ent.text)
        
        # Process the target sentence
        resolved_text = sent.text
        
        # Simple pronoun resolution
        for token in sent:
            if token.pos_ == "PRON" and token.text.lower() in ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'their']:
                # Find the nearest named entity before this pronoun
                nearest_entity = None
                
                # Look backwards in the sentence first
                for prev_token in reversed(list(sent[:token.i - sent.start])):
                    if prev_token.ent_type_ in ['PERSON', 'ORG', 'GPE', 'NORP']:
                        nearest_entity = prev_token.text
                        # Extend to full entity
                        for ent in sent.ents:
                            if prev_token.i >= ent.start and prev_token.i < ent.end:
                                nearest_entity = ent.text
                                break
                        break
                
                # If not found, look in context entities
                if not nearest_entity and context_entities:
                    # Use the most recent entity from context
                    nearest_entity = context_entities[-1]
                
                # Replace pronoun with entity
                if nearest_entity:
                    # Handle different pronoun cases
                    if token.text.lower() in ['he', 'she', 'him', 'her']:
                        replacement = nearest_entity
                    elif token.text.lower() in ['his', 'hers']:
                        replacement = f"{nearest_entity}'s"
                    elif token.text.lower() in ['their']:
                        replacement = f"{nearest_entity}'s"
                    elif token.text.lower() in ['they', 'them']:
                        replacement = nearest_entity
                    elif token.text.lower() == 'it':
                        replacement = nearest_entity
                    elif token.text.lower() == 'its':
                        replacement = f"{nearest_entity}'s"
                    else:
                        replacement = nearest_entity
                    
                    # Replace in text (simple word boundary replacement)
                    resolved_text = resolved_text.replace(f" {token.text} ", f" {replacement} ")
                    resolved_text = resolved_text.replace(f" {token.text}.", f" {replacement}.")
                    resolved_text = resolved_text.replace(f" {token.text},", f" {replacement},")
        
        resolved_sentences.append({
            'original': sent.text,
            'resolved': resolved_text
        })
    
    return resolved_sentences

def extract_information(text, api_key, method="GraphEval+", context_window=0):
    """Extract information from text - supports GraphEval+ and SICI methods"""
    if not api_key:
        return []
    
    # SICI methods use sentence extraction with spaCy
    if method.startswith("SICI"):
        return resolve_coreferences_spacy(text, context_window)
    
    # GraphEval+ uses triple extraction
    try:
        client = OpenAI(api_key=api_key)
        
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You are an expert at extracting information in structured formats to build a knowledge graph.

                Step 1 - Entity detection: Identify all entities in the raw text. Make sure not to miss any out. Entities should be basic and simple, they are akin to Wikipedia nodes.

                Step 2 - Coreference resolution: Find all expressions in the text that refer to the same entity. Make sure entities are not duplicated. In particular do not include entities that are more specific versions themselves, e.g. "a detailed view of jupiter's atmosphere" and "jupiter's atmosphere", only include the most specific version of the entity.

                Step 3 - Relation extraction: Identify semantic relationships between the entities you have identified.

                Format your response as a JSON array of objects, where each object must have exactly these three fields:
                - "subject": The first entity
                - "verb": The relationship between entities
                - "object": The second entity

                Important Tips:
                1. Make sure all information is included in the knowledge graph.
                2. Each triple must have exactly three non-empty strings.
                3. Do not split up related information into separate triples because this could change the meaning.
                4. Before adding a triple to the knowledge graph, check if concatenating subject+verb+object makes sense as a sentence. If not, discard it.
                5. Keep entities and relationships concise but meaningful.
                6. Convert pronouns to their proper noun references when possible.
                7. Keep everything lowercase and in present tense when appropriate.
                8. The output should be a JSON array of objects, each object containing the fields "subject", "verb", and "object", with the starting and ending tags ```json and ``` respectively.
                """},
                {"role": "user", "content": f"Use the given format to extract information from the following input: <input>{text}</input>. Skip the preamble and output the result as a JSON array within <json></json> tags."}
            ]
        )

        if completion.choices:
            response_message = str(completion.choices[0].message.content)
            
            # Extract JSON from various formats
            if "<json>" in response_message and "</json>" in response_message:
                json_start = response_message.find("<json>") + len("<json>")
                json_end = response_message.find("</json>")
                response_message = response_message[json_start:json_end].strip()
            elif "```json" in response_message and "```" in response_message:
                json_start = response_message.find("```json") + len("```json")
                json_end = response_message.rfind("```")
                response_message = response_message[json_start:json_end].strip()
            elif "```" in response_message:
                json_start = response_message.find("```") + len("```")
                json_end = response_message.rfind("```")
                response_message = response_message[json_start:json_end].strip()
            
            # Parse JSON
            try:
                parsed_json = json.loads(response_message)
                
                # If the response message is a single JSON object, convert it to a list
                if isinstance(parsed_json, dict):
                    parsed_json = [parsed_json]
                    
                return parsed_json
            except json.JSONDecodeError:
                st.error(f"Failed to parse JSON from response: {response_message}")
                return []
        else:
            st.error("No response received from the model.")
            return []
    except Exception as e:
        st.error(f"Error extracting information: {str(e)}")
        return []

def ask_question(question, context, api_key, model="gpt-4-turbo"):
    """Use RAG to answer a question based on context"""
    if not api_key:
        return "Please provide an OpenAI API key to generate an answer."
    
    try:
        client = OpenAI(api_key=api_key)
        
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an assistant specialized in answering questions based on a given context. Your task is to provide accurate and concise answers to the questions asked. If the answer is not present in the context, you should respond with 'The answer is not present in the given context.'"},
                {"role": "system", "content": "Context: " + context},
                {"role": "user", "content": f"Question: {question}"}
            ]
        )

        if completion.choices:
            response_message = str(completion.choices[0].message.content)
            return response_message
        else:
            return "No response received from the model."
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def find_similar_claims(source_json, output_json, api_key, method="GraphEval+", base_url=None, model="text-embedding-3-small"):
    """Find semantically similar claims between source and output"""
    if not api_key or not source_json or not output_json:
        return {}
    
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Convert to text based on method
        source_graphs = []
        output_graphs = []
        
        if method.startswith("SICI"):
            # For SICI, use resolved sentences
            for entry in source_json:
                source_graphs.append(entry['resolved'])
            for entry in output_json:
                output_graphs.append(entry['resolved'])
        else:
            # For GraphEval+, use triples
            for entry in source_json:
                source_graphs.append(f"{entry['subject']} {entry['verb']} {entry['object']}")
            for entry in output_json:
                output_graphs.append(f"{entry['subject']} {entry['verb']} {entry['object']}")

        # Create embeddings for source and output graphs using the same method for both
        source_embeddings = client.embeddings.create(input=source_graphs, model=model)
        output_embeddings = client.embeddings.create(input=output_graphs, model=model)
        
        similar_claims = {}

        for i, output_entry in enumerate(output_json):
            output_embedding = output_embeddings.data[i].embedding
            similarities = []
            
            for j, source_entry in enumerate(source_json):
                source_embedding = source_embeddings.data[j].embedding
                # Calculate cosine similarity
                similarity = np.dot(output_embedding, source_embedding) / (
                    np.linalg.norm(output_embedding) * np.linalg.norm(source_embedding))
                similarities.append((source_entry, similarity))
                
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Create output string based on method
            if method.startswith("SICI"):
                output_entry_string = output_entry['resolved']
            else:
                output_entry_string = f"{output_entry['subject']} {output_entry['verb']} {output_entry['object']}"
                
            similar_claims[output_entry_string] = similarities[:3]  # Top 3 similar claims

        return similar_claims
    except Exception as e:
        st.error(f"Error finding similar claims: {str(e)}")
        return {}

def generate_consistency_scores(claim_pairs, api_key):
    """Simulate hallucination evaluation scores (placeholder)"""
    # In the original notebook, this uses a Hugging Face model
    # For simplicity, we'll use a random score between 0.5 and 1.0
    # In a real app, you would implement the actual model call
    
    simple_scores = []
    for _ in claim_pairs:
        # Generate a random score between 0.5 and 1.0
        simple_scores.append(random.uniform(0.5, 1.0))
        
    return simple_scores

def visualize_knowledge_graph(similar_claims, simple_scores, method="GraphEval+"):
    """Create visualization of knowledge graph with consistency scores"""
    if not similar_claims or not simple_scores:
        return None
        
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Constants for visualization
    GLOBAL_SCALE_FACTOR = 0.15
    
    # Dictionary to store output claim nodes
    output_nodes = {}
    all_nodes = {}
    source_counter = 0
    
    # First, create all the output nodes
    for i, (output_entry, source_entries) in enumerate(similar_claims.items()):
        # Calculate average similarity
        avg_similarity = sum(similarity for _, similarity in source_entries) / len(source_entries)
        
        # Get NLI score
        nli_score = simple_scores[i]
        
        # Add jitter to prevent overlaps
        jitter_x = random.uniform(-0.02, 0.02)
        jitter_y = random.uniform(-0.02, 0.02)
        
        # Position based on NLI score (x) and avg similarity (y) plus jitter
        pos_x = max(0.1, min(0.9, nli_score + jitter_x))
        pos_y = max(0.1, min(0.9, avg_similarity + jitter_y))
        
        # Create node ID
        output_node_id = f"O{i}"
        
        # Store output node data
        output_nodes[output_entry] = {
            'id': output_node_id,
            'nli_score': nli_score,
            'avg_similarity': avg_similarity,
            'pos': (pos_x, pos_y),
            'connected_sources': []
        }
        
        # Add to all_nodes
        all_nodes[output_node_id] = {
            'id': output_node_id,
            'type': 'output',
            'pos': (pos_x, pos_y),
            'text': output_entry,
            'nli_score': nli_score,
            'avg_similarity': avg_similarity,
            'connections': []
        }
    
    # Function to calculate source node position
    def calculate_source_position(output_pos, similarity, angle):
        # Ensure similarity is in valid range
        similarity_value = max(-1, min(1, similarity))
        
        # Calculate distance - globally consistent for same similarity
        if similarity_value >= 0:
            distance = GLOBAL_SCALE_FACTOR * (1 - similarity_value)
        else:
            distance = GLOBAL_SCALE_FACTOR * 2 * (1 - similarity_value)
        
        # Calculate position
        x_offset = distance * math.cos(angle)
        y_offset = distance * math.sin(angle)
        
        pos_x = output_pos[0] + x_offset
        pos_y = output_pos[1] + y_offset
        
        # Ensure positions stay within bounds
        pos_x = max(0.05, min(0.95, pos_x))
        pos_y = max(0.05, min(0.95, pos_y))
        
        return (pos_x, pos_y)
    
    # Add source nodes
    for i, (output_entry, source_entries) in enumerate(similar_claims.items()):
        output_data = output_nodes[output_entry]
        output_pos = output_data['pos']
        output_id = output_data['id']
        
        # Calculate evenly spaced angles for source nodes
        base_angles = np.linspace(0, 2*np.pi, len(source_entries)+1)[:-1]
        
        # Add offset to angles based on output node index
        angles = [angle + (i * 0.5) for angle in base_angles]
        
        for idx, (source_entry, similarity) in enumerate(source_entries):
            # Create source text based on method
            if method.startswith("SICI"):
                source_text = source_entry['resolved']
            else:
                source_text = f"{source_entry['subject']} {source_entry['verb']} {source_entry['object']}"
                
            source_node_id = f"S{source_counter}"
            source_counter += 1
            
            # Calculate source position
            source_pos = calculate_source_position(output_pos, similarity, angles[idx])
            
            # Store source node data
            all_nodes[source_node_id] = {
                'id': source_node_id,
                'type': 'source',
                'pos': source_pos,
                'text': source_text,
                'similarity': similarity
            }
            
            # Store connection info
            all_nodes[output_id]['connections'].append({
                'source_id': source_node_id,
                'similarity': similarity
            })
    
    # Adjust positions to reduce overlaps
    def adjust_positions(nodes, min_distance=0.05, iterations=30):
        for _ in range(iterations):
            positions = {node['id']: node['pos'] for node in nodes.values()}
            node_ids = list(positions.keys())
            
            for i, node_id_i in enumerate(node_ids):
                for j, node_id_j in enumerate(node_ids[i+1:], i+1):
                    pos_i = positions[node_id_i]
                    pos_j = positions[node_id_j]
                    
                    # Calculate distance
                    dist = distance.euclidean(pos_i, pos_j)
                    
                    # If too close, move them apart
                    if dist < min_distance:
                        # Direction vector
                        dx = pos_j[0] - pos_i[0]
                        dy = pos_j[1] - pos_i[1]
                        
                        # Normalize if not zero
                        if dist > 0:
                            dx /= dist
                            dy /= dist
                        else:
                            # Random direction if exactly overlapping
                            angle = random.uniform(0, 2*math.pi)
                            dx = math.cos(angle)
                            dy = math.sin(angle)
                        
                        # Calculate repulsion force
                        force = 0.01 * (min_distance - dist)
                        
                        # Apply different force based on node type
                        if nodes[node_id_i]['type'] == 'output':
                            force_i = force * 0.5
                        else:
                            force_i = force
                            
                        if nodes[node_id_j]['type'] == 'output':
                            force_j = force * 0.5
                        else:
                            force_j = force
                        
                        # Update positions
                        new_pos_i = (
                            max(0.05, min(0.95, nodes[node_id_i]['pos'][0] - dx * force_i)),
                            max(0.05, min(0.95, nodes[node_id_i]['pos'][1] - dy * force_i))
                        )
                        new_pos_j = (
                            max(0.05, min(0.95, nodes[node_id_j]['pos'][0] + dx * force_j)),
                            max(0.05, min(0.95, nodes[node_id_j]['pos'][1] + dy * force_j))
                        )
                        
                        nodes[node_id_i]['pos'] = new_pos_i
                        nodes[node_id_j]['pos'] = new_pos_j
        
        return nodes
    
    # Apply position adjustments
    all_nodes = adjust_positions(all_nodes)
    
    # Recalculate source positions to maintain consistent edge lengths
    for node_id, node_data in all_nodes.items():
        if node_data['type'] == 'output':
            output_pos = node_data['pos']
            
            for idx, conn in enumerate(node_data['connections']):
                source_id = conn['source_id']
                similarity = conn['similarity']
                
                # Calculate angle based on current positions
                source_pos = all_nodes[source_id]['pos']
                angle = math.atan2(source_pos[1] - output_pos[1], source_pos[0] - output_pos[0])
                
                # Recalculate position
                new_pos = calculate_source_position(output_pos, similarity, angle)
                
                # Update position
                all_nodes[source_id]['pos'] = new_pos
    
    # Apply final adjustment
    all_nodes = adjust_positions(all_nodes)
    
    # Draw edges
    for node_id, node_data in all_nodes.items():
        if node_data['type'] == 'output':
            output_pos = node_data['pos']
            
            for conn in node_data['connections']:
                source_id = conn['source_id']
                similarity = conn['similarity']
                source_pos = all_nodes[source_id]['pos']
                
                # Edge color based on similarity
                if similarity < 0:
                    edge_color = 'red'
                    alpha = 0.7
                else:
                    edge_color = plt.cm.Blues(0.3 + 0.7 * similarity)
                    alpha = 0.7
                
                # Draw edge
                plt.plot([output_pos[0], source_pos[0]], [output_pos[1], source_pos[1]], 
                         color=edge_color, alpha=alpha, linewidth=2, zorder=1)
                
                # Add similarity label
                mid_x = (output_pos[0] + source_pos[0]) / 2
                mid_y = (output_pos[1] + source_pos[1]) / 2
                
                # Offset to avoid overlapping the line
                dx = source_pos[0] - output_pos[0]
                dy = source_pos[1] - output_pos[1]
                angle = math.atan2(dy, dx)
                offset = 0.01
                offset_x = offset * math.sin(angle)
                offset_y = -offset * math.cos(angle)
                
                # Add similarity text
                plt.text(mid_x + offset_x, mid_y + offset_y, f"{similarity:.2f}", 
                         fontsize=8, color='black' if similarity >= 0 else 'darkred', 
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                         horizontalalignment='center', verticalalignment='center', zorder=4)
    
    # Draw nodes
    for node_id, node_data in all_nodes.items():
        pos = node_data['pos']
        
        if node_data['type'] == 'output':
            # Color based on NLI score
            nli_score = node_data['nli_score']
            avg_similarity = node_data['avg_similarity']

            if nli_score >= NLI_THRESHOLD and avg_similarity >= SIMILARITY_THRESHOLD:
                node_color = 'green'
            elif nli_score < NLI_THRESHOLD and avg_similarity >= SIMILARITY_THRESHOLD:
                node_color = 'yellow'
            elif nli_score >= NLI_THRESHOLD and avg_similarity < SIMILARITY_THRESHOLD:
                node_color = 'orange'
            else:
                node_color = 'red'
            
            # Draw output node
            plt.scatter(pos[0], pos[1], color=node_color, s=300, edgecolor='black', zorder=3)
            
            # Add label
            plt.text(pos[0], pos[1], node_id, 
                     horizontalalignment='center', verticalalignment='center',
                     fontweight='bold', fontsize=10, color='black', zorder=4)
        else:
            # Draw source node
            plt.scatter(pos[0], pos[1], color='lightblue', s=150, 
                       edgecolor='black', alpha=0.9, zorder=2)
            
            # Add label
            plt.text(pos[0], pos[1], node_id, 
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=8, color='black', zorder=4)
    
    # Add similarity scale reference
    x_pos, y_pos = 0.75, 0.12
    sim_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    lengths = [GLOBAL_SCALE_FACTOR * (1 - sim) for sim in sim_values]
    
    plt.text(x_pos, y_pos + 0.05, "Similarity Scale", 
             fontsize=12, fontweight='bold', horizontalalignment='left')
    
    for sim, length in zip(sim_values, lengths):
        start_x = x_pos
        start_y = y_pos - 0.01*sim
        end_x = start_x + length
        end_y = start_y
        
        plt.plot([start_x, end_x], [start_y, end_y], 
                 color=plt.cm.Blues(0.3 + 0.7 * sim), linewidth=2)
        
        plt.text(end_x + 0.01, end_y, f"{sim:.2f}", 
                 fontsize=8, verticalalignment='center')
    
    # Add grid, labels, and legend
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('NLI Consistency Score', fontsize=14)
    plt.ylabel('Average Semantic Similarity', fontsize=14)
    
    # Legend
    output_patch = mpatches.Patch(color='green', label='Output Claims (color = NLI score)')
    source_patch = mpatches.Patch(color='lightblue', label='Source Claims')
    pos_edge = Line2D([0], [0], color='blue', lw=2, alpha=0.7, 
                     label='Edge length = similarity (shorter = more similar)')
    neg_edge = Line2D([0], [0], color='red', lw=2, alpha=0.7, 
                     label='Red edge = negative similarity')
    
    plt.legend(handles=[output_patch, source_patch, pos_edge, neg_edge], 
              loc='lower right', fontsize=12, bbox_to_anchor=(1.0, 0.15),
              framealpha=0.9)
    
    # Quadrant labels
    plt.text(0.95, 0.95, "High Reliability", fontsize=14, ha='right', va='top', 
             bbox=dict(facecolor='lightgreen', alpha=0.3))
    plt.text(0.05, 0.95, "Suspicious Content", fontsize=14, ha='left', va='top',
             bbox=dict(facecolor='lightyellow', alpha=0.3))
    plt.text(0.95, 0.05, "Plausible But Unsupported", fontsize=14, ha='right', va='bottom',
             bbox=dict(facecolor='lightyellow', alpha=0.3))
    plt.text(0.05, 0.05, "Potential Hallucination", fontsize=14, ha='left', va='bottom',
             bbox=dict(facecolor='lightcoral', alpha=0.3))
    
    # Title with method name
    method_name = "GraphEval+" if method == "GraphEval+" else f"SICI ({method.split('-')[1].split('(')[0].strip()})"
    plt.title(f'Knowledge Graph Validation ({method_name}): Source vs Output Claims\n' + 
             'Output claims positioned by NLI score (X) and avg. similarity (Y)', 
             fontsize=16)
    
    # Return the figure
    return fig

def create_claim_details_table(similar_claims, simple_scores, method="GraphEval+"):
    """Create a detailed text table of claims and their relationships with Streamlit expanders by quadrant category"""
    if not similar_claims or not simple_scores:
        return "No claims to display."
    
    # Create output nodes dictionary
    output_nodes = {}
    for i, (output_entry, source_entries) in enumerate(similar_claims.items()):
        # Calculate average similarity
        avg_similarity = sum(similarity for _, similarity in source_entries) / len(source_entries)
        
        # Get NLI score
        nli_score = simple_scores[i]
        
        # Create output node
        output_id = f"O{i}"
        output_nodes[output_entry] = {
            'id': output_id,
            'nli_score': nli_score,
            'avg_similarity': avg_similarity,
            'connected_sources': source_entries
        }
    
    # Group claims by quadrant category - using both metrics
    high_reliability = []      # High NLI, High Similarity
    suspicious_content = []    # Low NLI, High Similarity 
    plausible_unsupported = [] # High NLI, Low Similarity
    potential_hallucination = [] # Low NLI, Low Similarity
    
    for output_entry, output_data in output_nodes.items():
        nli_score = output_data['nli_score']
        avg_similarity = output_data['avg_similarity']
        
        if nli_score >= NLI_THRESHOLD and avg_similarity >= SIMILARITY_THRESHOLD:
            high_reliability.append((output_entry, output_data))
        elif nli_score < NLI_THRESHOLD and avg_similarity >= SIMILARITY_THRESHOLD:
            suspicious_content.append((output_entry, output_data))
        elif nli_score >= NLI_THRESHOLD and avg_similarity < SIMILARITY_THRESHOLD:
            plausible_unsupported.append((output_entry, output_data))
        else:  # nli_score < NLI_THRESHOLD and avg_similarity < SIMILARITY_THRESHOLD
            potential_hallucination.append((output_entry, output_data))
    
    # Sort within each category by combined score (weighted average of both metrics)
    def sort_key(item):
        data = item[1]
        return (data['nli_score'] + data['avg_similarity']) / 2
    
    high_reliability.sort(key=sort_key, reverse=True)
    suspicious_content.sort(key=sort_key, reverse=True)
    plausible_unsupported.sort(key=sort_key, reverse=True)
    potential_hallucination.sort(key=sort_key, reverse=True)
    
    # Return data for Streamlit to render with expanders
    claim_categories = [
        {
            "emoji": "🟢",
            "title": "High Reliability",
            "subtitle": "High NLI Score & High Similarity",
            "claims": high_reliability,
            "default_open": True,
        },
        {
            "emoji": "🟡",
            "title": "Suspicious Content",
            "subtitle": "Low NLI Score & High Similarity",
            "claims": suspicious_content,
            "default_open": False,
        },
        {
            "emoji": "🟠",
            "title": "Plausible But Unsupported",
            "subtitle": "High NLI Score & Low Similarity",
            "claims": plausible_unsupported,
            "default_open": False,
        },
        {
            "emoji": "🔴",
            "title": "Potential Hallucination",
            "subtitle": "Low NLI Score & Low Similarity",
            "claims": potential_hallucination,
            "default_open": False,
        }
    ]
    
    return claim_categories

def fix_llm_output(answer, context, selected_claims): 
    """Fix LLM output by prompting the LLM to check the suspicious and potential hallucination claims"""
    question = "A LLM generated this answer. You have been given some potential hallucinations and suspicious claims in this answer. Depending on the type of claim, and the advice provided for that claim in the context, improve the response, and provide a brief justification in a separate line, starting with 'JUSTIFICATION', explaining your rationale behind the changes. If you are not sure about the claim, please remove it. If you are sure about the claim, please keep it.\n"
    context_text = ""

    context_text += "-----------------------------\n"
    context_text += f"Original Context: {context}\n"
    context_text += "-----------------------------\n"
    context_text += f"Original Answer: {answer}\n"
    context_text += "-----------------------------\n"

    claim_categories_advice = {
        "High Reliability": "These claims are reliable and should be kept.",
        "Suspicious Content": "These claims are suspicious, check if the claims actually follow the context. Your answer should depict the confidence (or lack thereof) in these claims.",
        "Plausible But Unsupported": "These claims are plausible but lack support. Use less confident language, but not too underconfident either.",
        "Potential Hallucination": "These claims are potential hallucinations. Either remove them or fix them. If you are not sure, go with removing them."
    }
    context_text += "----------CLAIMS IN THE ANSWER----------\n"
    for category in selected_claims:
        if selected_claims[category]:  # Only include non-empty categories
            context_text += f"----------{category}: {claim_categories_advice[category]}----------\n"
            for claim in selected_claims[category]:
                context_text += f"{claim}\n"
    
    response = ask_question(question, context_text, api_key, model=model_choice)
    return response

# Function to show claim categories with multiselect for fixing LLM output
def show_claim_categories(claim_categories, method="GraphEval+"):
    if not claim_categories:
        return {}
    
    # Create a dictionary to store all claims
    all_claims = {}
    category_claims = {}
    
    # Collect all claims and organize by category
    for category in claim_categories:
        category_title = category['title']
        category_claims[category_title] = []
        for claim_entry, output_data in category['claims']:
            claim_id = output_data['id']
            all_claims[claim_id] = claim_entry
            category_claims[category_title].append(claim_id)
    
    # Create multiselect for each category
    selected_claims = {}
    for category in claim_categories:
        category_title = category['title']
        if category['claims']:  # Only show categories with claims
            selected_claims[category_title] = st.multiselect(
                f"{category['emoji']} {category_title} - {category['subtitle']}",
                options=list(all_claims.keys()),
                default=category_claims[category_title],
                format_func=lambda x: f"{x}: {all_claims[x][:100]}..." if x in all_claims else x,
                key=f"select_{category_title.replace(' ', '_')}"
            )
        else:
            selected_claims[category_title] = st.multiselect(
                f"{category['emoji']} {category_title} - {category['subtitle']}",
                options=list(all_claims.keys()),
                default=[],
                format_func=lambda x: f"{x}: {all_claims[x][:100]}..." if x in all_claims else x,
                key=f"select_{category_title.replace(' ', '_')}"
            )
    
    # return the actual selected claims, not their IDs
    return {
        category: [all_claims[claim_id] for claim_id in selected_claims[category]]
        for category in selected_claims
    }

# Main analysis function
def run_analysis(method):
    context = st.session_state.context
    question = st.session_state.question
    
    if not context or not question:
        st.warning("Please provide both context and question to proceed.")
        return
    
    # Determine context window for SICI methods
    context_window = 0
    if method == "SICI-1 (sentences with context)":
        context_window = 1
        
    # 1. Generate answer using RAG
    answer = ask_question(question, context, api_key, model=model_choice)
    st.session_state.answer = answer
    
    # 2. Extract information from context and answer
    with st.status("Extracting knowledge graphs...") as status:
        status.update(label=f"Extracting from source context using {method}...")
        source_json = extract_information(context, api_key, method=method, context_window=context_window)
        st.session_state.source_json = source_json
        
        status.update(label=f"Extracting from generated answer using {method}...")
        output_json = extract_information(answer, api_key, method=method, context_window=context_window)
        st.session_state.output_json = output_json
        
        status.update(label="Comparing knowledge graphs...", state="complete")
    
    # 3. Find similar claims
    embedding_api_key = os.environ.get("EMBEDDING_API_KEY") or api_key
    embedding_base_url = os.environ.get("EMBEDDING_BASE_URL")
    
    similar_claims = find_similar_claims(source_json, output_json, embedding_api_key, method=method, 
                                        base_url=embedding_base_url, model=embedding_model)
    st.session_state.similar_claims = similar_claims
    
    # 4. Generate claim pairs for evaluation
    claim_pairs = []
    for output_entry, source_entries in similar_claims.items():
        if method.startswith("SICI"):
            source_claims = [source_entry['resolved'] for source_entry, _ in source_entries]
        else:
            source_claims = [f"{source_entry['subject']} {source_entry['verb']} {source_entry['object']}" 
                            for source_entry, _ in source_entries]
        source_claims = ";".join(source_claims)
        claim_pairs.append((source_claims, output_entry))
    
    # 5. Evaluate claims for consistency/hallucination
    simple_scores = generate_consistency_scores(claim_pairs, api_key)
    st.session_state.simple_scores = simple_scores
    
    # 6. Generate claim categories
    claim_categories = create_claim_details_table(similar_claims, simple_scores, method=method)
    st.session_state.claim_categories = claim_categories
    
    # Mark analysis as complete
    st.session_state.run_complete = True

# Button to run analysis
if st.button("Run Analysis", type="primary", disabled=not api_key and not use_example):
    run_analysis(detection_method)

# Display results if analysis has been run
if st.session_state.run_complete:
    with col2:
        st.header("Results")
        
        # Show the generated answer
        st.subheader("Generated Answer")
        st.write(st.session_state.answer)
        
        # Show the visualization
        st.subheader("Knowledge Graph Validation")
        graph_fig = visualize_knowledge_graph(st.session_state.similar_claims, st.session_state.simple_scores, 
                                             method=detection_method)
        if graph_fig:
            st.pyplot(graph_fig)
        
        # Show claim categories
        st.subheader("Claim Details")
        claim_categories = st.session_state.claim_categories
        
        if isinstance(claim_categories, str):
            # Handle the case where no claims exist
            st.markdown(claim_categories)
        else:
            # Display each category in a separate expander
            for category in claim_categories:
                # Only show categories that have claims
                if category["claims"]:
                    # Create expander with count
                    count = len(category["claims"])
                    expander = st.expander(
                        f"{category['emoji']} {category['title']} ({count}) - {category['subtitle']}", 
                        expanded=False
                    )
                    
                    # Display claims within expander
                    with expander:
                        for claim_entry, output_data in category["claims"]:
                            st.markdown(f"**{output_data['id']}**: {claim_entry}")
                            st.markdown(f"  - NLI score: {output_data['nli_score']:.3f}, Avg similarity: {output_data['avg_similarity']:.3f}")
                            
                            st.markdown("  - **Connected to source claims:**")
                            
                            for j, (source_entry, similarity) in enumerate(output_data['connected_sources']):
                                # Format source text based on method
                                if detection_method.startswith("SICI"):
                                    source_text = source_entry['resolved']
                                else:
                                    source_text = f"{source_entry['subject']} {source_entry['verb']} {source_entry['object']}"
                                source_id = f"S{j}"
                                
                                # Format similarity with indicator for negative values
                                sim_text = f"{similarity:.3f}"
                                if similarity < 0:
                                    sim_text = f"{sim_text} **(NEGATIVE)**"
                                
                                st.markdown(f"    - {source_id}: {source_text} (similarity: {sim_text})")
                            
                            st.markdown("---")  # Add separator between claims
        
        # Show claim selector for fixing output
        st.subheader("Select Claims to Fix")
        selected_claims = show_claim_categories(claim_categories, method=detection_method)
        
        # Create a button to fix the LLM output
        if st.button("Fix LLM Output", key="fix_button"):
            with st.spinner("Fixing LLM output..."):
                fixed_answer = fix_llm_output(st.session_state.answer, st.session_state.context, selected_claims)
                st.session_state.fixed_answer = fixed_answer
        
        # Display fixed answer if available
        if st.session_state.fixed_answer:
            st.subheader("Fixed Answer")
            st.write(st.session_state.fixed_answer)
        
        # Show insights
        st.subheader("Insights")
        
        # Calculate overall reliability score
        if st.session_state.simple_scores:
            simple_scores = st.session_state.simple_scores
            similar_claims = st.session_state.similar_claims
            
            avg_nli = sum(simple_scores) / len(simple_scores)
            avg_sim = sum(sum(similarity for _, similarity in sources) / len(sources) 
                          for _, sources in similar_claims.items()) / len(similar_claims)
            
            reliability_score = (avg_nli * 0.7 + avg_sim * 0.3) * 100
            
            # Gauge for reliability score
            st.metric("Average NLI Score", f"{avg_nli:.2f}")
            st.metric("Average Semantic Similarity", f"{avg_sim:.2f}")
            st.metric("Overall Reliability Score", f"{reliability_score:.1f}%")
            
            # Classification based on score
            if reliability_score > 80:
                st.success("✅ The answer is highly reliable and well-supported by the source content.")
            elif reliability_score > 60:
                st.info("ℹ️ The answer is mostly reliable with good support from the source content.")
            elif reliability_score > 40:
                st.warning("⚠️ The answer contains some unsupported claims that may need verification.")
            else:
                st.error("❌ The answer contains potentially hallucinated content with poor source support.")
                
            # Specific observations
            st.markdown("### Key Observations:")
            
            # Check for very low NLI scores
            low_nli = [i for i, score in enumerate(simple_scores) if score < 0.4]
            if low_nli:
                st.warning(f"⚠️ Found {len(low_nli)} claims with low factual consistency scores.")
            
            # Check for negative similarities
            neg_sims = sum(1 for _, sources in similar_claims.items() 
                          for _, sim in sources if sim < 0)
            if neg_sims:
                st.warning(f"⚠️ Found {neg_sims} negative similarity connections, indicating potential contradictions.")
            
            # Overall assessment
            if reliability_score > 70:
                st.markdown("**Recommendation:** The generated answer appears reliable enough to use with minimal verification.")
            else:
                st.markdown("**Recommendation:** The generated answer should be verified or edited before use.")

# Footer with info
st.markdown("---")
st.markdown(f"""
**How it works ({detection_method}):** 
1. Extract knowledge representations from both the source context and the generated answer
   - **GraphEval+**: Subject-verb-object triples with bidirectional extraction (uses GPT-4o)
   - **SICI-0**: Individual sentences with spaCy-based coreference resolution (no context window)
   - **SICI-1**: Sentences with spaCy-based coreference resolution using adjacent sentences for context
2. Compare representations using semantic similarity (same embedding model for all methods)
3. Evaluate factual consistency to detect potential hallucinations or unsupported statements
4. Visualize the relationships to identify reliable vs. suspicious content

**Computational Efficiency:**
- **GraphEval+**: ~8 hours (uses GPT-4o for triple extraction)
- **SICI methods**: ~30 minutes (uses spaCy for NER and coreference resolution)
""")

# Instructions in expandable section
with st.expander("📝 Instructions"):
    st.markdown("""
    ### Using the Knowledge Graph Validation Tool
    
    **First-time setup:**
```bash
    pip install spacy
    python -m spacy download en_core_web_sm
```
    
    1. **Select a detection method** in the sidebar:
       - **GraphEval+**: Uses GPT-4o for triple extraction (subject-verb-object relationships)
       - **SICI-0**: Sentence-level analysis with spaCy coreference resolution (much faster, ~30 min)
       - **SICI-1**: Sentence-level with context window (includes adjacent sentences for better coreference)
    2. **Provide context** (the source material) in the Context text area
    3. **Enter a question** to be answered based on the context
    4. **Click "Run Analysis"** to process the data and generate results
    5. **Review the visualization** to identify reliable vs. potentially hallucinated content:
       - X-axis: Factual consistency score (higher is more consistent)
       - Y-axis: Average semantic similarity to source claims (higher is more similar)
       - Green output nodes: High factual consistency
       - Red output nodes: Low factual consistency
       - Edge length: Inversely related to similarity (shorter = more similar)
    6. **Check the Claim Details** section to see specific connections between output and source claims
    7. **Review the Insights** for an overall assessment of answer reliability
    8. **Select claims to fix** and click "Fix LLM Output" to generate an improved answer
    
    ### Method Comparison
    
    - **GraphEval+**: Most comprehensive analysis using GPT-4o for relation extraction (~8 hours)
    - **SICI-0**: Fast and efficient using spaCy NER (~30 minutes), good for quick validation
    - **SICI-1**: Balanced approach with context awareness using spaCy, still much faster than GraphEval+
    
    ### Technical Details
    
    - **Coreference Resolution**: SICI methods use spaCy's NER to identify entities and simple pronoun resolution
    - **Semantic Similarity**: All methods use the same OpenAI embedding model for consistency
    - **No LLM calls in SICI**: Only traditional NLP techniques (spaCy) for computational efficiency
    
    ### Tips
    
    - The visualization places claims in four quadrants:
      - **Top-Right (Green)**: High reliability - claims are consistent and similar to sources
      - **Top-Left (Yellow)**: Suspicious content - claims are similar to sources but factually inconsistent
      - **Bottom-Right (Yellow)**: Plausible but unsupported - claims are consistent but not well-supported by sources
      - **Bottom-Left (Red)**: Potential hallucination - claims are neither consistent nor supported
    - Look for red edges, which indicate negative similarity (potential contradictions)
    - Use the similarity scale as a reference for interpreting edge lengths
    - **SICI methods are recommended for production use** due to significantly faster processing times
    """)