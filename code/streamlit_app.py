import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

# Configure Streamlit page
st.set_page_config(
    page_title="H&M Markdown Prediction & What-If Simulator",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File paths
project_root = Path(__file__).parent.parent
data_dir = project_root / "data"

@st.cache_data
def load_sample_data():
    """Load sample H&M data for the demo"""
    
    # Load articles data
    articles_path = data_dir / "raw" / "articles.csv"
    if articles_path.exists():
        articles = pd.read_csv(articles_path, nrows=1000)  # Sample for demo
    else:
        # Create synthetic sample data
        np.random.seed(42)
        articles = pd.DataFrame({
            'article_id': range(100001, 100101),
            'product_type_name': np.random.choice(['T-shirt', 'Dress', 'Trousers', 'Sweater', 'Jacket'], 100),
            'product_group_name': np.random.choice(['Garment Upper body', 'Garment Lower body', 'Garment Full body'], 100),
            'colour_group_name': np.random.choice(['Black', 'White', 'Blue', 'Red', 'Grey'], 100),
            'department_name': np.random.choice(['Menswear', 'Womenwear', 'Kids'], 100),
            'price': np.random.uniform(10, 150, 100).round(2)
        })
    
    # Add synthetic features for markdown prediction
    articles['current_price'] = articles.get('price', np.random.uniform(10, 150, len(articles)))
    articles['sales_velocity'] = np.random.uniform(0.1, 5.0, len(articles))
    articles['inventory_level'] = np.random.randint(10, 1000, len(articles))
    articles['days_since_launch'] = np.random.randint(1, 365, len(articles))
    articles['markdown_risk_score'] = np.random.uniform(0.05, 0.85, len(articles))
    
    return articles

def calculate_markdown_probability(current_price, price_change_pct, product_type, days_since_launch, sales_velocity):
    """Simulate markdown probability calculation"""
    
    # Base probability
    base_prob = 0.15
    
    # Price increase reduces markdown risk
    price_factor = max(0, price_change_pct / 100) * -0.3
    
    # Product type factors
    type_factors = {
        'Dress': 0.05, 'T-shirt': -0.02, 'Trousers': -0.01, 
        'Sweater': 0.03, 'Jacket': 0.02
    }
    type_factor = type_factors.get(product_type, 0)
    
    # Age factor (older items more likely to be marked down)
    age_factor = min(days_since_launch / 365 * 0.2, 0.2)
    
    # Sales velocity factor (slow movers more likely to be marked down)
    velocity_factor = max(0, (2.0 - sales_velocity) * 0.1)
    
    # Calculate final probability
    final_prob = base_prob + price_factor + type_factor + age_factor + velocity_factor
    return np.clip(final_prob, 0.01, 0.95)

def calculate_financial_impact(current_price, new_price, markdown_prob, inventory_level, sales_velocity):
    """Calculate financial impact of pricing decisions"""
    
    # Price elasticity simulation
    price_change = (new_price - current_price) / current_price
    demand_response = -1.2 * price_change  # Elastic demand
    
    # Sales projection
    baseline_sales = sales_velocity * 30  # 30-day projection
    new_sales = baseline_sales * (1 + demand_response)
    
    # Revenue calculations
    baseline_revenue = baseline_sales * current_price
    new_revenue = new_sales * new_price
    
    # Markdown cost (if item gets marked down later)
    markdown_discount = 0.25  # Assume 25% markdown
    markdown_cost = markdown_prob * inventory_level * current_price * markdown_discount
    
    # Net impact
    revenue_impact = new_revenue - baseline_revenue
    risk_adjusted_impact = revenue_impact - markdown_cost
    
    return {
        'baseline_revenue': baseline_revenue,
        'new_revenue': new_revenue,
        'revenue_impact': revenue_impact,
        'markdown_cost': markdown_cost,
        'risk_adjusted_impact': risk_adjusted_impact,
        'projected_sales': new_sales
    }

# Main app
def main():
    st.title("🏷️ H&M Markdown Prediction & What-If Simulator")
    st.markdown("**Predict markdown risk and simulate pricing scenarios with financial impact analysis**")
    
    # Load data
    with st.spinner("Loading H&M product data..."):
        articles_df = load_sample_data()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Simulation Controls")
    
    # Product selection
    st.sidebar.subheader("Product Selection")
    
    # Filter by product type
    product_types = ['All'] + sorted(articles_df['product_type_name'].unique().tolist())
    selected_product_type = st.sidebar.selectbox("Product Type", product_types)
    
    if selected_product_type != 'All':
        filtered_df = articles_df[articles_df['product_type_name'] == selected_product_type]
    else:
        filtered_df = articles_df
    
    # Select specific article
    article_options = filtered_df['article_id'].tolist()
    selected_article = st.sidebar.selectbox(
        "Article ID", 
        article_options,
        format_func=lambda x: f"{x} ({filtered_df[filtered_df['article_id']==x]['product_type_name'].iloc[0]})"
    )
    
    # Get selected article data
    article_data = filtered_df[filtered_df['article_id'] == selected_article].iloc[0]
    
    # What-if scenario inputs
    st.sidebar.subheader("📊 What-If Scenario")
    
    current_price = article_data['current_price']
    price_change = st.sidebar.slider(
        "Price Change (%)", 
        min_value=-50, max_value=50, value=0, step=5,
        help="Adjust price by percentage"
    )
    
    new_price = current_price * (1 + price_change / 100)
    st.sidebar.metric("New Price", f"${new_price:.2f}", f"{price_change:+}%")
    
    # Promotion settings
    st.sidebar.subheader("🎯 Promotion Settings")
    is_promotion = st.sidebar.checkbox("Apply Promotion")
    if is_promotion:
        promo_duration = st.sidebar.slider("Promotion Duration (days)", 1, 30, 14)
        promo_boost = st.sidebar.slider("Sales Boost (%)", 10, 100, 25)
    else:
        promo_duration = 0
        promo_boost = 0
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Product Information")
        
        # Product details
        product_info = {
            "Article ID": article_data['article_id'],
            "Product Type": article_data['product_type_name'],
            "Color": article_data['colour_group_name'],
            "Department": article_data['department_name'],
            "Current Price": f"${article_data['current_price']:.2f}",
            "Inventory Level": f"{article_data['inventory_level']:,} units",
            "Days Since Launch": f"{article_data['days_since_launch']} days",
            "Sales Velocity": f"{article_data['sales_velocity']:.1f} units/day"
        }
        
        for key, value in product_info.items():
            st.metric(key, value)
    
    with col2:
        st.subheader("⚠️ Markdown Risk Analysis")
        
        # Calculate markdown probability
        markdown_prob = calculate_markdown_probability(
            current_price, price_change, 
            article_data['product_type_name'],
            article_data['days_since_launch'],
            article_data['sales_velocity']
        )
        
        # Risk level
        if markdown_prob < 0.2:
            risk_level = "Low"
            risk_color = "green"
        elif markdown_prob < 0.5:
            risk_level = "Medium"
            risk_color = "orange"
        else:
            risk_level = "High"
            risk_color = "red"
        
        st.metric(
            "Markdown Risk", 
            f"{markdown_prob:.1%}",
            help="Probability of markdown in next 30 days"
        )
        
        st.markdown(f"**Risk Level:** :{risk_color}[{risk_level}]")
        
        # Risk factors
        st.markdown("**Key Risk Factors:**")
        factors = []
        if article_data['days_since_launch'] > 180:
            factors.append("• Product age (>6 months)")
        if article_data['sales_velocity'] < 1.0:
            factors.append("• Low sales velocity")
        if price_change < -10:
            factors.append("• Significant price reduction")
        
        if factors:
            for factor in factors:
                st.markdown(factor)
        else:
            st.markdown("• No major risk factors identified")
    
    # Financial Impact Analysis
    st.subheader("💰 Financial Impact Analysis")
    
    # Calculate financial metrics
    impact = calculate_financial_impact(
        current_price, new_price, markdown_prob,
        article_data['inventory_level'], article_data['sales_velocity']
    )
    
    # Apply promotion boost if selected
    if is_promotion:
        impact['projected_sales'] *= (1 + promo_boost / 100)
        impact['new_revenue'] = impact['projected_sales'] * new_price
        impact['revenue_impact'] = impact['new_revenue'] - impact['baseline_revenue']
        impact['risk_adjusted_impact'] = impact['revenue_impact'] - impact['markdown_cost']
    
    # Display metrics in columns
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric(
            "Revenue Impact (30d)",
            f"${impact['revenue_impact']:,.0f}",
            delta=f"{(impact['revenue_impact']/impact['baseline_revenue'])*100:+.1f}%"
        )
    
    with metric_col2:
        st.metric(
            "Projected Sales",
            f"{impact['projected_sales']:.0f} units",
            delta=f"{((impact['projected_sales']/max(article_data['sales_velocity']*30, 1))-1)*100:+.1f}%"
        )
    
    with metric_col3:
        st.metric(
            "Markdown Risk Cost",
            f"${impact['markdown_cost']:,.0f}",
            help="Expected cost if item gets marked down"
        )
    
    with metric_col4:
        color = "normal"
        if impact['risk_adjusted_impact'] > 0:
            color = "normal"
        elif impact['risk_adjusted_impact'] < -1000:
            color = "inverse"
        
        st.metric(
            "Net Impact",
            f"${impact['risk_adjusted_impact']:,.0f}",
            help="Revenue impact minus markdown risk"
        )
    
    # Scenario comparison chart
    st.subheader("📈 Scenario Comparison")
    
    # Create comparison scenarios
    scenarios = []
    price_changes = [-20, -10, 0, 10, 20]
    
    for pc in price_changes:
        scenario_price = current_price * (1 + pc / 100)
        scenario_prob = calculate_markdown_probability(
            current_price, pc, article_data['product_type_name'],
            article_data['days_since_launch'], article_data['sales_velocity']
        )
        scenario_impact = calculate_financial_impact(
            current_price, scenario_price, scenario_prob,
            article_data['inventory_level'], article_data['sales_velocity']
        )
        
        scenarios.append({
            'Price Change (%)': pc,
            'New Price ($)': scenario_price,
            'Markdown Risk (%)': scenario_prob * 100,
            'Net Impact ($)': scenario_impact['risk_adjusted_impact'],
            'Revenue Impact ($)': scenario_impact['revenue_impact']
        })
    
    scenario_df = pd.DataFrame(scenarios)
    
    # Plot scenarios
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=scenario_df['Price Change (%)'],
        y=scenario_df['Net Impact ($)'],
        mode='lines+markers',
        name='Net Impact',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=scenario_df['Price Change (%)'],
        y=scenario_df['Revenue Impact ($)'],
        mode='lines+markers',
        name='Revenue Impact',
        line=dict(color='green', width=2),
        marker=dict(size=6)
    ))
    
    # Highlight current scenario (find closest match if exact not found)
    current_matches = scenario_df[scenario_df['Price Change (%)'] == price_change]
    if len(current_matches) > 0:
        current_idx = current_matches.index[0]
        fig.add_trace(go.Scatter(
            x=[price_change],
            y=[scenario_df.iloc[current_idx]['Net Impact ($)']],
            mode='markers',
            name='Current Scenario',
            marker=dict(color='red', size=12, symbol='star')
        ))
    else:
        # If exact match not found, find closest
        closest_idx = (scenario_df['Price Change (%)'] - price_change).abs().idxmin()
        fig.add_trace(go.Scatter(
            x=[scenario_df.iloc[closest_idx]['Price Change (%)']],
            y=[scenario_df.iloc[closest_idx]['Net Impact ($)']],
            mode='markers',
            name='Current Scenario (closest)',
            marker=dict(color='orange', size=12, symbol='star')
        ))
    
    fig.update_layout(
        title="Financial Impact vs Price Change",
        xaxis_title="Price Change (%)",
        yaxis_title="Financial Impact ($)",
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("🎯 Recommendations")
    
    optimal_scenario = scenario_df.loc[scenario_df['Net Impact ($)'].idxmax()]
    
    recommendations = []
    
    if optimal_scenario['Net Impact ($)'] > impact['risk_adjusted_impact']:
        recommendations.append(
            f"💡 **Consider {optimal_scenario['Price Change (%)']:+.0f}% price change** "
            f"for ${optimal_scenario['Net Impact ($)']:,.0f} additional net impact"
        )
    
    if markdown_prob > 0.5:
        recommendations.append("⚠️ **High markdown risk detected** - Consider proactive pricing or promotion")
    
    if article_data['sales_velocity'] < 1.0:
        recommendations.append("📉 **Low sales velocity** - Marketing push or price reduction may be needed")
    
    if not recommendations:
        recommendations.append("✅ **Current pricing appears optimal** - No immediate action needed")
    
    for rec in recommendations:
        st.markdown(rec)
    
    # Export results
    st.subheader("📤 Export Results")
    
    if st.button("Generate Action List"):
        action_list = pd.DataFrame([{
            'Article ID': selected_article,
            'Product Type': article_data['product_type_name'],
            'Current Price': current_price,
            'Recommended Action': f"{optimal_scenario['Price Change (%)']:+.0f}% price change",
            'Expected Net Impact': optimal_scenario['Net Impact ($)'],
            'Markdown Risk': f"{markdown_prob:.1%}",
            'Priority': 'High' if markdown_prob > 0.5 else 'Medium' if markdown_prob > 0.3 else 'Low'
        }])
        
        csv = action_list.to_csv(index=False)
        st.download_button(
            label="Download Action List CSV",
            data=csv,
            file_name=f"markdown_action_list_{selected_article}.csv",
            mime="text/csv"
        )
        
        st.success("Action list generated successfully!")

if __name__ == "__main__":
    main()
