import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data_processor import DataProcessor
from src.model_trainer import GECToRTrainer
from src.evaluator import ModelEvaluator
from src.utils import set_seeds, create_transformation_tags

# Set page config
st.set_page_config(
    page_title="GECToR Vehicle Typo Correction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None

def main():
    st.title("🚗 GECToR Vehicle Information Typo Correction")
    st.markdown("### Fine-tune Transformer Model for Vehicle Data Typo Detection and Correction")
    
    # Sidebar for navigation
    st.sidebar.title("ML Pipeline Steps")
    step = st.sidebar.selectbox(
        "Choose ML Step:",
        ["1. Data Exploration", "2. Data Preprocessing", "3. Model Training", 
         "4. Model Evaluation", "5. Predictions & Export"]
    )
    
    # Set random seeds for reproducibility
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, max_value=9999)
    set_seeds(seed)
    
    if step == "1. Data Exploration":
        data_exploration()
    elif step == "2. Data Preprocessing":
        data_preprocessing()
    elif step == "3. Model Training":
        model_training()
    elif step == "4. Model Evaluation":
        model_evaluation()
    elif step == "5. Predictions & Export":
        predictions_and_export()

def data_exploration():
    st.header("📊 Data Exploration")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Vehicle Dataset CSV", 
        type=['csv'],
        help="Upload the CSV file containing vehicle information with typos"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.session_state.raw_data = df
            st.session_state.data_loaded = True
            
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                error_rate = (df['error_type'] != 'correct').sum() / len(df) * 100
                st.metric("Error Rate", f"{error_rate:.1f}%")
            
            # Display sample data
            st.subheader("Sample Data")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data statistics
            st.subheader("Data Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Error Type Distribution**")
                error_counts = df['error_type'].value_counts()
                fig = px.bar(
                    x=error_counts.index, 
                    y=error_counts.values,
                    title="Distribution of Error Types"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Brand Distribution**")
                brand_counts = df['expected_brand'].value_counts()
                fig = px.pie(
                    values=brand_counts.values, 
                    names=brand_counts.index,
                    title="Vehicle Brand Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed error analysis
            st.subheader("Error Analysis")
            
            # Multiple error types
            multi_errors = df[df['error_type'].str.contains(',', na=False)]
            st.write(f"**Records with Multiple Errors:** {len(multi_errors)}")
            
            if len(multi_errors) > 0:
                st.dataframe(multi_errors.head(), use_container_width=True)
            
            # Year distribution
            st.subheader("Year Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    df, x='expected_year', 
                    title="Expected Year Distribution",
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Invalid years analysis
                invalid_years = df[df['user_input_year'].isin([1800, 1970, 2099])]
                st.write(f"**Invalid Years Count:** {len(invalid_years)}")
                if len(invalid_years) > 0:
                    invalid_year_counts = invalid_years['user_input_year'].value_counts()
                    fig = px.bar(
                        x=invalid_year_counts.index,
                        y=invalid_year_counts.values,
                        title="Invalid Year Values"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to begin data exploration")

def data_preprocessing():
    st.header("⚙️ Data Preprocessing")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Exploration step")
        return
    
    df = st.session_state.raw_data
    
    # Initialize data processor
    if st.session_state.data_processor is None:
        st.session_state.data_processor = DataProcessor()
    
    processor = st.session_state.data_processor
    
    st.subheader("Data Processing Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Set Size", 0.1, 0.3, 0.2, 0.05)
        val_size = st.slider("Validation Set Size", 0.1, 0.3, 0.15, 0.05)
    
    with col2:
        max_length = st.number_input("Max Sequence Length", 32, 256, 128, 16)
        tag_vocab_size = st.number_input("Tag Vocabulary Size", 1000, 10000, 5000, 500)
    
    if st.button("🔄 Process Data", type="primary"):
        with st.spinner("Processing data..."):
            try:
                # Process the data
                processed_data = processor.process_data(
                    df, 
                    test_size=test_size, 
                    val_size=val_size,
                    max_length=max_length,
                    tag_vocab_size=tag_vocab_size
                )
                
                st.session_state.processed_data = processed_data
                
                st.success("✅ Data processing completed!")
                
                # Display processing results
                st.subheader("Processing Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training Samples", len(processed_data['train_data']))
                with col2:
                    st.metric("Validation Samples", len(processed_data['val_data']))
                with col3:
                    st.metric("Test Samples", len(processed_data['test_data']))
                
                # Show sample processed data
                st.subheader("Sample Processed Data")
                sample_idx = 0
                sample = processed_data['train_data'][sample_idx]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Input Tokens:**")
                    st.text(" ".join(sample['input_tokens']))
                    st.write("**Target Tokens:**")
                    st.text(" ".join(sample['target_tokens']))
                
                with col2:
                    st.write("**Transformation Tags:**")
                    st.text(" ".join(sample['tags']))
                    st.write("**Input IDs Shape:**")
                    st.text(f"{sample['input_ids'].shape}")
                
                # Transformation statistics
                st.subheader("Transformation Statistics")
                all_tags = []
                for data_split in ['train_data', 'val_data', 'test_data']:
                    for sample in processed_data[data_split]:
                        all_tags.extend(sample['tags'])
                
                tag_counts = pd.Series(all_tags).value_counts().head(20)
                
                fig = px.bar(
                    x=tag_counts.values,
                    y=tag_counts.index,
                    orientation='h',
                    title="Top 20 Transformation Tags"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error processing data: {str(e)}")
                st.exception(e)

def model_training():
    st.header("🤖 Model Training")
    
    if not hasattr(st.session_state, 'processed_data'):
        st.warning("⚠️ Please process data first in the Data Preprocessing step")
        return
    
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        learning_rate = st.number_input("Learning Rate", 1e-6, 1e-3, 5e-5, format="%.0e")
        batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
        num_epochs = st.number_input("Number of Epochs", 1, 10, 3)
    
    with col2:
        warmup_steps = st.number_input("Warmup Steps", 0, 1000, 100)
        weight_decay = st.number_input("Weight Decay", 0.0, 0.1, 0.01)
        gradient_clipping = st.number_input("Gradient Clipping", 0.1, 2.0, 1.0)
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        model_name = st.text_input("Model Name", "gotutiyan/gector-roberta-base-5k")
        save_steps = st.number_input("Save Every N Steps", 100, 1000, 500)
        eval_steps = st.number_input("Evaluate Every N Steps", 50, 500, 200)
    
    if st.button("🚀 Start Training", type="primary"):
        # Initialize trainer
        if st.session_state.trainer is None:
            st.session_state.trainer = GECToRTrainer(model_name=model_name)
        
        trainer = st.session_state.trainer
        
        # Training configuration
        training_config = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'warmup_steps': warmup_steps,
            'weight_decay': weight_decay,
            'gradient_clipping': gradient_clipping,
            'save_steps': save_steps,
            'eval_steps': eval_steps
        }
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Training metrics placeholders
        col1, col2 = st.columns(2)
        with col1:
            loss_chart = st.empty()
        with col2:
            acc_chart = st.empty()
        
        try:
            # Train the model
            training_history = trainer.train(
                st.session_state.processed_data,
                training_config,
                progress_callback=lambda progress, status, metrics: update_training_ui(
                    progress, status, metrics, progress_bar, status_text, loss_chart, acc_chart
                )
            )
            
            st.session_state.training_history = training_history
            st.session_state.model_trained = True
            
            st.success("🎉 Training completed successfully!")
            
            # Final training summary
            st.subheader("Training Summary")
            final_metrics = training_history['final_metrics']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Final Training Loss", f"{final_metrics['train_loss']:.4f}")
            with col2:
                st.metric("Final Validation Loss", f"{final_metrics['val_loss']:.4f}")
            with col3:
                st.metric("Final Validation Accuracy", f"{final_metrics['val_accuracy']:.4f}")
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            st.exception(e)

def update_training_ui(progress, status, metrics, progress_bar, status_text, loss_chart, acc_chart):
    """Update training UI with progress and metrics"""
    progress_bar.progress(progress)
    status_text.text(status)
    
    if metrics and len(metrics['train_loss']) > 0:
        # Update loss chart
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(
            y=metrics['train_loss'], 
            name='Training Loss', 
            line=dict(color='blue')
        ))
        if 'val_loss' in metrics and len(metrics['val_loss']) > 0:
            loss_fig.add_trace(go.Scatter(
                y=metrics['val_loss'], 
                name='Validation Loss', 
                line=dict(color='red')
            ))
        loss_fig.update_layout(title="Training Loss", xaxis_title="Step", yaxis_title="Loss")
        loss_chart.plotly_chart(loss_fig, use_container_width=True)
        
        # Update accuracy chart
        if 'val_accuracy' in metrics and len(metrics['val_accuracy']) > 0:
            acc_fig = go.Figure()
            acc_fig.add_trace(go.Scatter(
                y=metrics['val_accuracy'], 
                name='Validation Accuracy', 
                line=dict(color='green')
            ))
            acc_fig.update_layout(title="Validation Accuracy", xaxis_title="Step", yaxis_title="Accuracy")
            acc_chart.plotly_chart(acc_fig, use_container_width=True)

def model_evaluation():
    st.header("📈 Model Evaluation")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(st.session_state.trainer.model, st.session_state.trainer.tokenizer)
    
    st.subheader("Model Performance Metrics")
    
    # Evaluate on test set
    if st.button("🧪 Evaluate on Test Set", type="primary"):
        with st.spinner("Evaluating model..."):
            try:
                test_data = st.session_state.processed_data['test_data']
                evaluation_results = evaluator.evaluate(test_data)
                
                st.session_state.evaluation_results = evaluation_results
                
                # Display overall metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Overall Accuracy", f"{evaluation_results['overall_accuracy']:.4f}")
                with col2:
                    st.metric("Token Accuracy", f"{evaluation_results['token_accuracy']:.4f}")
                with col3:
                    st.metric("Sequence Accuracy", f"{evaluation_results['sequence_accuracy']:.4f}")
                with col4:
                    st.metric("F1 Score", f"{evaluation_results['f1_score']:.4f}")
                
                # Error type analysis
                st.subheader("Performance by Error Type")
                error_metrics = evaluation_results['error_type_metrics']
                
                metrics_df = pd.DataFrame(error_metrics).T
                st.dataframe(metrics_df, use_container_width=True)
                
                # Confusion matrix
                st.subheader("Confusion Matrix")
                if 'confusion_matrix' in evaluation_results:
                    cm = evaluation_results['confusion_matrix']
                    fig = px.imshow(
                        cm, 
                        title="Confusion Matrix",
                        color_continuous_scale="Blues",
                        aspect="auto"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Sample predictions
                st.subheader("Sample Predictions")
                sample_predictions = evaluation_results['sample_predictions'][:10]
                
                for i, pred in enumerate(sample_predictions):
                    with st.expander(f"Sample {i+1}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Input:**")
                            st.text(" ".join(pred['input_tokens']))
                            st.write("**Expected:**")
                            st.text(" ".join(pred['expected_tokens']))
                        with col2:
                            st.write("**Predicted:**")
                            st.text(" ".join(pred['predicted_tokens']))
                            st.write("**Correct:**")
                            st.text("✅" if pred['correct'] else "❌")
                
            except Exception as e:
                st.error(f"❌ Evaluation failed: {str(e)}")
                st.exception(e)
    
    # Training history visualization
    if st.session_state.training_history:
        st.subheader("Training History")
        
        history = st.session_state.training_history
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Loss plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history['train_loss'], 
                name='Training Loss',
                line=dict(color='blue')
            ))
            if 'val_loss' in history:
                fig.add_trace(go.Scatter(
                    y=history['val_loss'], 
                    name='Validation Loss',
                    line=dict(color='red')
                ))
            fig.update_layout(title="Training Loss Over Time", xaxis_title="Step", yaxis_title="Loss")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Accuracy plot
            if 'val_accuracy' in history:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history['val_accuracy'], 
                    name='Validation Accuracy',
                    line=dict(color='green')
                ))
                fig.update_layout(title="Validation Accuracy Over Time", xaxis_title="Step", yaxis_title="Accuracy")
                st.plotly_chart(fig, use_container_width=True)

def predictions_and_export():
    st.header("🔮 Predictions & Model Export")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first")
        return
    
    # Interactive prediction
    st.subheader("Interactive Typo Correction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        plate = st.text_input("License Plate", "ABC 1234")
        brand = st.text_input("Brand", "Toyot")
        model = st.text_input("Model", "Camr")
        year = st.number_input("Year", 1990, 2030, 2021)
    
    with col2:
        if st.button("🔧 Correct Typos", type="primary"):
            try:
                # Prepare input
                input_text = f"{plate} {brand} {model} {year}"
                
                # Get prediction
                evaluator = ModelEvaluator(st.session_state.trainer.model, st.session_state.trainer.tokenizer)
                corrected_text = evaluator.predict_single(input_text)
                
                st.success("✅ Correction completed!")
                
                st.write("**Original:**")
                st.text(input_text)
                st.write("**Corrected:**")
                st.text(corrected_text)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
    
    # Batch prediction
    st.subheader("Batch Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload CSV for batch correction", 
        type=['csv'],
        help="Upload a CSV file with columns: user_input_plate, user_input_brand, user_input_model, user_input_year"
    )
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("**Sample of uploaded data:**")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            if st.button("🚀 Process Batch", type="primary"):
                with st.spinner("Processing batch predictions..."):
                    evaluator = ModelEvaluator(st.session_state.trainer.model, st.session_state.trainer.tokenizer)
                    
                    # Process each row
                    predictions = []
                    for _, row in batch_df.iterrows():
                        input_text = f"{row['user_input_plate']} {row['user_input_brand']} {row['user_input_model']} {row['user_input_year']}"
                        corrected = evaluator.predict_single(input_text)
                        predictions.append(corrected)
                    
                    batch_df['corrected_text'] = predictions
                    
                    st.success("✅ Batch processing completed!")
                    st.dataframe(batch_df, use_container_width=True)
                    
                    # Download results
                    csv = batch_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"corrected_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Batch processing failed: {str(e)}")
    
    # Model export
    st.subheader("Model Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_path = st.text_input("Export Path", "fine_tuned_gector_model")
        
        if st.button("💾 Export Model", type="primary"):
            try:
                with st.spinner("Exporting model..."):
                    # Save model and tokenizer
                    trainer = st.session_state.trainer
                    trainer.save_model(export_path)
                    
                    # Save training configuration and results
                    export_info = {
                        'model_name': trainer.model_name,
                        'training_config': st.session_state.training_history.get('config', {}),
                        'final_metrics': st.session_state.training_history.get('final_metrics', {}),
                        'export_timestamp': datetime.now().isoformat()
                    }
                    
                    with open(f"{export_path}/export_info.json", 'w') as f:
                        json.dump(export_info, f, indent=2)
                    
                    st.success(f"✅ Model exported to: {export_path}")
                    
                    # Create download link for the export info
                    st.download_button(
                        label="📄 Download Export Info",
                        data=json.dumps(export_info, indent=2),
                        file_name="model_export_info.json",
                        mime="application/json"
                    )
            
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
    
    with col2:
        st.info("""
        **Export includes:**
        - Fine-tuned model weights
        - Tokenizer configuration
        - Training configuration
        - Performance metrics
        - Export metadata
        """)

if __name__ == "__main__":
    main()
