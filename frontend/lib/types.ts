export type PlayerProjection = {
  player_id: number;
  name: string;
  team_id: number;
  position: string;
  price: number;
  predicted_points: number;
  ci_low: number;
  ci_high: number;
  variance_proxy: number;
  value_vs_price: number;
};

export type TradeRecommendation = {
  trade_out_id: number;
  trade_in_id: number;
  expected_value_gain: number;
  risk_team_delta_std: number;
  net_budget_delta: number;
  mc_mean: number;
  mc_std: number;
};
